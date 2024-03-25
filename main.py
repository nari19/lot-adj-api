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
params = {
    "0.70": 1.3,
    "0.60": 1.2,
    "0.55": 1.1,
    "0.50": 1.0,
    "0.45": 0.2,
    "0.40": 0.1,
}

# Define request body data model
class InputData(BaseModel):
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
    
    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        # 予測確率がparamsのkeyの値を超えたらparamsのvalueを出力
        lot = 1.0
        for key in params.keys():
            if prediction > float(key):
                lot = params[key]
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
