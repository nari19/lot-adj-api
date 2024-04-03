from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import lightgbm as lgb
import pandas as pd
import os

# Define your FastAPI app
app = FastAPI()

# Lot Adjustments Parameters
lotParams = {
    "0.52": 1.0,
    "0.00": 0.1,
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

    # Day, Hour, Minuteを削除
    input_df = input_df.drop(["Day", "Hour", "Minute"], axis=1)
    
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

# Define root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the lightGBM prediction API!"}

# This part is for local testing using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
