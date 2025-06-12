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

# Define your FastAPI app
app = FastAPI()

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
            last_row['scaled_Close']
        ]
        
        # ONNXモデルのURLを構築
        model_url = f"https://storage.googleapis.com/model-cnd/20250601212746/FX/{ohlc_data.symbol}/5M/close.onnx"
        
        # モデルをダウンロード
        response = requests.get(model_url)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail=f"モデルが見つかりません: {ohlc_data.symbol}")
        
        # モデルを読み込む
        session = ort.InferenceSession(response.content)
        
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
        
        # エントリー判定
        entry_signal = 1 if abs(deviation) > threshold else 0
        
        result = {
            "symbol": ohlc_data.symbol,
            "deviation": deviation,
            "last_close": last_close,
            "predicted_price": actual_prediction,
            "threshold": threshold,
            "entry_signal": entry_signal
        }
        print(f"Prediction for {ohlc_data.symbol}: \n"
              f"Last Close: {last_close}, \n"
              f"Predicted Price: {actual_prediction}, \n"
              f"Deviation: {deviation}%, \n"
              f"Threshold: {threshold}, \n"
              f"Entry Signal: {entry_signal}")
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
    return df

# This part is for local testing using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
