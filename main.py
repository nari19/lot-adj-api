from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import requests
import datetime
import pytz
import investpy
import cachetools
from cachetools import TTLCache

# Define your FastAPI app
app = FastAPI()

# Lot Adjustments Parameters
lotParams = {
    "0.0": 0.8,
    "-1000": 0.8,
}

# Define the list of symbols
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

# Define request body data model
class InputData(BaseModel):
    Symbol: str
    BuySell: str
    Indi1: int
    Indi2: int
    Param1: int
    Param2: int
    Day: int
    Hour: int
    Minute: int
    RSI1: float
    RSI2: float
    RSI3: float
    RSI4: float
    RSI5: float
    MACD1: float
    MACD_Sig1: float
    MACD2: float
    MACD_Sig2: float
    MACD3: float
    MACD_Sig3: float
    MACD4: float
    MACD_Sig4: float
    MACD5: float
    MACD_Sig5: float

# Load your pre-trained lightGBM model
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    try:
        model = lgb.Booster(model_file=model_path)
        return model
    except Exception as e:
        raise Exception(f"Failed to load model from '{model_path}': {str(e)}")

model_path = "predictionModel.pkl.txt"
model = load_model(model_path)

# Define prediction endpoint
@app.post("/predict/")
async def predict(data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # "BuySell"を数値に変換 (Buy: 1, Sell: 0)
    input_df["BuySell"] = input_df["BuySell"].apply(lambda x: 1 if x == "Buy" else 0)
    # "Symbol"を数値に変換 (EURJPY: 0, GBPJPY: 1, USDJPY: 2, ...)
    input_df["Symbol"] = input_df["Symbol"].apply(lambda x: symbols.index(x))

    # Minuteカラムをsin, cosに変換させて、Minute_sin, Minute_cosに格納
    input_df['Minute_sin'] = np.sin(2 * np.pi * input_df['Minute'] / 60)
    input_df['Minute_cos'] = np.cos(2 * np.pi * input_df['Minute'] / 60)
    input_df = input_df.drop(['Minute'], axis=1)
    # Hourカラムをsin, cosに変換させて、Hour_sin, Hour_cosに格納
    input_df['Hour_sin'] = np.sin(2 * np.pi * input_df['Hour'] / 24)
    input_df['Hour_cos'] = np.cos(2 * np.pi * input_df['Hour'] / 24)
    input_df = input_df.drop(['Hour'], axis=1)
    
    # input_dfからDayを削除
    input_df = input_df.drop(columns=['Day'])
    
    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        # 予測確率がparamsのkeyの値を超えたらparamsのvalueを出力
        lot = 1.0
        for key in lotParams.keys():
            if prediction > float(key):
                lot = lotParams[key]
                break
        return {"prediction": prediction, "lot": lot}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

# Define root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the lightGBM prediction API!"}

# This part is for local testing using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
