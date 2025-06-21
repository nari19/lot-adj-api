from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import os
import requests
import datetime
import pytz
import investpy
import cachetools
from cachetools import TTLCache
import onnxruntime as ort
from pydantic import BaseModel
from typing import List, Dict
import json
import pandas_ta as ta

# Define your FastAPI app
app = FastAPI()

# モデルキャッシュをグローバル変数として定義
# 120kb x 21銘柄 = 約2.5MB
model_cache = {}

symbols = [
    "EURJPY", "GBPJPY", "USDJPY", "CADJPY", "NZDJPY", "CHFJPY",
    "USDCHF", "EURCHF", "CADCHF", "NZDCHF", "GBPCHF",
    "GBPCAD", "USDCAD", "EURCAD", "NZDCAD",
    "EURUSD", "NZDUSD", "GBPUSD",
    "EURNZD", "GBPNZD",
    "EURGBP"
]

# Cache for params endpoint
cache = TTLCache(maxsize=1, ttl=20)

# Cache for predict endpoint
predict_cache = TTLCache(maxsize=30, ttl=20)  # 最大30個のシンボルの結果をキャッシュ

# Define root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the lightGBM prediction API!"}


# Define endpoint for getting parameters
@app.get("/params/")
async def get_params():
    # Check if the result is cached
    if 'params' in cache:
        return cache['params']

    symbol_relations = {
        "euro zone": ["EUR", "GBP", "CHF"],
        "united states": ["USD", "CAD"],
        "japan": ["JPY"],
        "united kingdom": ["GBP"],
        "switzerland": ["CHF"],
        "canada": ["CAD"],
        "new zealand": ["NZD"],
        "australia": ["NZD"],
    }

    # https://nari19.github.io/s-params/params.txt からパラメータを取得
    params_url = "https://nari19.github.io/s-params/params.txt"
    params = requests.get(params_url).text

    # 現在時刻を取得
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

    # 現在時刻からfrom_date, to_dateを取得
    # from_date: 現在の日付, to_date: 現在の日付 + 1日
    from_date = now.strftime('%d/%m/%Y')
    to_date = (now + datetime.timedelta(days=1)).strftime('%d/%m/%Y')

    economic_data = investpy.economic_calendar(
        time_zone='GMT +9:00',
        # symbol_relationsのkeyを配列で取得
        countries=list(symbol_relations.keys()),
        from_date=from_date,
        to_date=to_date,
        importances=['high']
    )
    # economic_dataがemptyの場合は、paramsをreturnして終了
    if economic_data.empty:
        print("economic_data is empty")
        cache['params'] = params
        return params

    # timeが"All Day", currencyがNone, importanceがNoneのいずれかに当てはまる行を削除
    economic_data = economic_data[
        (economic_data['time'] != 'All Day') &
        (economic_data['currency'].notnull()) &
        (economic_data['importance'].notnull())
    ]

    # 現在日時から-3時間、+3時間の間にある経済指標を取得
    from_hour = 2
    to_hour = 5
    # 現在日時から-X時間したdateとtimeを取得
    from_datetime = (now - datetime.timedelta(hours=from_hour)).strftime('%Y-%m-%d %H:%M:%S')
    print("from_datetime: ", from_datetime)

    # 現在日時から+X時間したdateとtimeを取得
    to_datetime = (now + datetime.timedelta(hours=to_hour)).strftime('%Y-%m-%d %H:%M:%S')
    print("to_datetime: ", to_datetime)

    # 経済指標の時間がfrom_datetimeからto_datetimeの間にあるものを取得
    economic_data = economic_data[
        (pd.to_datetime(economic_data['date'] + ' ' + economic_data['time'], format='%d/%m/%Y %H:%M') >= pd.to_datetime(from_datetime)) &
        (pd.to_datetime(economic_data['date'] + ' ' + economic_data['time'], format='%d/%m/%Y %H:%M') <= pd.to_datetime(to_datetime))
    ]
    print(economic_data)

    # economic_data['currency']を元にsymbolを取得
    target_symbols = []
    for zone in economic_data['zone']:
        for symbol, currencies in symbol_relations.items():
            if zone in symbol:
                target_symbols.append(currencies)
                break

    # target_symbolsをフラット
    target_symbols = [symbol for symbols in target_symbols for symbol in symbols]
    # target_symbolsを重複削除
    target_symbols = list(set(target_symbols))
    print(target_symbols)

    # paramsをテーブル形式に変換
    # header: symbol, indi1, indi2, param1, param2
    params = params.split("\n")

    # paramsのループを回し、target_symbolsに含まれる行は{symbol}, 0, 0, 0, 0に変換させる
    for i, param in enumerate(params):
        symbol = param.split(",")[0]
        symbols = [symbol[:3], symbol[3:]]
        # symbolsのどちらかがtarget_symbolsに含まれているか確認
        if any([s in target_symbols for s in symbols]):
            params[i] = f"{symbol}, 0, 0, 0, 0"

    # paramsを改行で結合
    params = "\n".join(params)
    print(params + "\n")

    # Cache the result
    cache['params'] = params

    return params


class OHLCData(BaseModel):
    symbol: str
    data: List[Dict[str, float]]

@app.post("/predict/")
async def predict_deviation(ohlc_data: OHLCData):
    try:
        # キャッシュキーの生成（シンボル名）
        cache_key = ohlc_data.symbol
        
        # 予測結果のキャッシュチェック
        if cache_key in predict_cache:
            print(f"Returning cached result for {cache_key}")
            return predict_cache[cache_key]
            
        # 入力データをDataFrameに変換
        df = pd.DataFrame(ohlc_data.data)
        
        # 必要なカラムが存在するか確認
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail="必要なカラム（open, high, low, close）が不足しています")
        
        # データが500行以上あるか確認
        if len(df) < 500:
            raise HTTPException(status_code=400, detail="データは500行以上必要です")
        
        # テクニカル指標を計算
        df = calculate_technical_indicators(df)
        
        # NaNを含む行を削除
        df = df.dropna()

        # 最後の行の特徴量を取得
        last_row = df.iloc[-1]
        features = [
            last_row['time_sin'],
            last_row['time_cos'],
            last_row['day_of_week_sin'],
            last_row['day_of_week_cos'],
            last_row['is_tokyo_session'],
            last_row['is_london_session'],
            last_row['is_newyork_session'],
            last_row['200_High'],
            last_row['200_Low'],
            last_row['75_SMA'],
            last_row['75_High'],
            last_row['75_Low'],
            last_row['20_SMA'],
            last_row['20_High'],
            last_row['20_Low'],
            last_row['5_SMA'],
            last_row['5_High'],
            last_row['5_Low'],
            last_row['scaled_High'],
            last_row['scaled_Low'],
            last_row['scaled_Open'],
            last_row['scaled_Close'],
            last_row['RSI_14'],
            last_row['MACD_12_26_9'],
            last_row['MACDh_12_26_9'],
            last_row['MACDs_12_26_9'],
            last_row['BBL_20_2.0'],
            last_row['BBM_20_2.0'],
            last_row['BBU_20_2.0'],
            last_row['BBB_20_2.0'],
            last_row['BBP_20_2.0'],
            last_row['ATRr_14']
        ]
        
        # モデルのキャッシュチェック
        if cache_key not in model_cache:
            model_url = f"https://storage.googleapis.com/model-cnd/20250620212746/FX/{ohlc_data.symbol}/5M/close.onnx"
            response = requests.get(model_url)
            if response.status_code != 200:
                raise HTTPException(status_code=404, detail=f"モデルが見つかりません: {ohlc_data.symbol}")
            model_cache[cache_key] = ort.InferenceSession(response.content)
        
        # キャッシュされたモデルを使用
        session = model_cache[cache_key]
        
        # 予測を実行
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        prediction = session.run([output_name], {input_name: np.array([features], dtype=np.float32)})[0][0][0]
        
        # 予測値を実際の価格に変換
        last_close = df['close'].iloc[-1]
        last_200sma = df['200SMA'].iloc[-1]
        actual_prediction = last_200sma * (1 + prediction)
        
        # 乖離率を計算
        deviation = (actual_prediction - last_close) / last_close * 100
        
        # 閾値パラメータを取得
        threshold_params = get_threshold_params()
        threshold = threshold_params.get(ohlc_data.symbol, 0.0)
        
        # GMT+2/GMT+3タイムゾーンの現在時刻を取得
        helsinki_tz = pytz.timezone('Europe/Helsinki')
        current_helsinki_time = datetime.datetime.now(helsinki_tz)
        current_hour = current_helsinki_time.hour
        
        # エントリー判定
        entry_signal = 1 if abs(deviation) > threshold else 0
        
        # 時間制限によるエントリー制御
        # GMT+2/GMT+3タイムゾーンで22:00から24:59以外の時間帯ではエントリーしない
        if not ((current_hour >= 22 and current_hour <= 23) or (current_hour >= 0 and current_hour < 1)):
            entry_signal = 0
        
        result = {
            "symbol": ohlc_data.symbol,
            "deviation": deviation,
            "last_close": last_close,
            "predicted_price": actual_prediction,
            "threshold": threshold,
            "entry_signal": entry_signal
        }
        print(f"Prediction for {ohlc_data.symbol}: \n"
              f"Current Helsinki Time: {current_helsinki_time.strftime('%Y-%m-%d %H:%M:%S %Z')}, \n"
              f"Last Close: {last_close}, \n"
              f"Predicted Price: {actual_prediction}, \n"
              f"Deviation: {deviation}%, \n"
              f"Threshold: {threshold}, \n"
              f"Entry Signal: {entry_signal}")
              
        # 結果をキャッシュに保存
        predict_cache[cache_key] = result
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_threshold_params():
    """閾値パラメータを取得する関数"""
    params_url = "https://nari19.github.io/s-params/lgbm/params.txt"
    response = requests.get(params_url)
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="閾値パラメータが見つかりません")
    
    params = {}
    for line in response.text.strip().split('\n'):
        symbol, threshold = line.split(', ')
        params[symbol] = float(threshold)
    return params


def calculate_technical_indicators(df):
    """テクニカル指標を計算"""
    # 200SMAを計算
    df['200SMA'] = df['close'].rolling(window=200).mean()
    df['5SMA_Close'] = df['close'].rolling(window=5).mean()
    df['5SMA_High'] = df['high'].rolling(window=5).mean()
    df['5SMA_Low'] = df['low'].rolling(window=5).mean()
    # 時間特徴量の計算
    df = calculate_time_features(df)
    # 各期間のSMA（単純移動平均）を計算
    TECHNICAL_PERIODS = [200, 75, 20, 5]
    for period in TECHNICAL_PERIODS:
        # 200SMAに対する変化率で計算
        df[f'{period}_SMA'] = (df['close'].rolling(window=period).mean() - df['200SMA']) / df['200SMA']
        df[f'{period}_High'] = (df['high'].rolling(window=period).max() - df['200SMA']) / df['200SMA']
        df[f'{period}_Low'] = (df['low'].rolling(window=period).min() - df['200SMA']) / df['200SMA']
    # 価格データも200SMAに対する変化率に変換
    df['scaled_High'] = (df['high'] - df['200SMA']) / df['200SMA']
    df['scaled_Low'] = (df['low'] - df['200SMA']) / df['200SMA']
    df['scaled_Open'] = (df['open'] - df['200SMA']) / df['200SMA']
    df['scaled_Close'] = (df['close'] - df['200SMA']) / df['200SMA']
    # pandas_taを使用してテクニカル指標を追加
    #   'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9',
    #   'MACDs_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0',
    #   'BBP_20_2.0', 'ATRr_14'
    df.ta.rsi(close=df['close'], length=14, append=True)
    df.ta.macd(close=df['close'], fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(close=df['close'], length=20, std=2, append=True)
    df.ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14, append=True)
    return df


def calculate_time_features(df):
    """時間特徴量を計算する関数"""

    # タイムゾーン'Europe/Helsinki' (GMT+2/GMT+3)の現在の日時をdatetime型で取得
    helsinki_tz = pytz.timezone('Europe/Helsinki')
    current_time = datetime.datetime.now(helsinki_tz)
    
    # DataFrameのインデックスを現在時刻から生成（データの長さ分）
    local_time = pd.date_range(start=current_time, periods=len(df), freq='5T', tz=helsinki_tz)
        
    # --- タイムゾーン変換 ---
    # 'Europe/Helsinki' (GMT+2/GMT+3)のタイムゾーンで取得した時刻をUTCに変換することで、夏時間・冬時間も自動で考慮される
    utc_time = local_time.tz_convert('UTC')

    # 時間と分を抽出 (UTC時間から)
    hour_utc = utc_time.hour
    minute_utc = utc_time.minute
    
    # 時間と分を組み合わせた総分数を計算（0-1439分）
    total_minutes_utc = hour_utc * 60 + minute_utc
    
    # 時間のsin/cos変換（UTC時間の24時間周期）
    df['time_sin'] = np.sin(2 * np.pi * total_minutes_utc / 1440)
    df['time_cos'] = np.cos(2 * np.pi * total_minutes_utc / 1440)
    
    # 曜日の特徴量（sin/cos変換）- UTC基準
    day_of_week = utc_time.dayofweek # 月曜日=0, 日曜日=6
    df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    # --- 主要市場の取引時間帯フラグ (UTC基準) ---
    # UTCに変換した時刻を元に、各市場のコアタイムを判定
    df['is_tokyo_session'] = ((hour_utc >= 0) & (hour_utc < 9)).astype(int)
    df['is_london_session'] = ((hour_utc >= 7) & (hour_utc < 16)).astype(int) # ロンドン夏時間も考慮
    df['is_newyork_session'] = ((hour_utc >= 12) & (hour_utc < 21)).astype(int) # NY夏時間も考慮

    return df

# This part is for local testing using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
