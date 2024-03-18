from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import lightgbm as lgb
import pandas as pd

# Define your FastAPI app
app = FastAPI()

# Load your pre-trained lightGBM model
model_path = "path_to_your_pretrained_model"  # Update with your model path
model = lgb.Booster(model_file=model_path)

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

# Define prediction endpoint
@app.post("/predict/")
async def predict(data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        return {"prediction": prediction}
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
