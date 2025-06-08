import pandas as pd
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from contextlib import asynccontextmanager
import joblib

# SETUP & CONFIGURATION 
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "flight-delay-regressor"
MODEL_STAGE = "Production"
PREPROCESSOR_PATH = "models/preprocessor.joblib"

# Initialize placeholders
model = None
preprocessor = None

# LIFESPAN EVENT HANDLER 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the ML model and preprocessor at application startup.
    """
    global model, preprocessor
    print("Application startup: Loading model and preprocessor...")
    try:
        # Load Model from MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Model '{REGISTERED_MODEL_NAME}/{MODEL_STAGE}' loaded successfully.")

        # Load Preprocessor from file
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"✅ Preprocessor loaded successfully from {PREPROCESSOR_PATH}.")

    except Exception as e:
        print(f"🛑 Error loading model or preprocessor: {e}")
        model = None
        preprocessor = None
    
    yield
    
    # Shutdown Logic
    print("Application shutdown.")
    model = None
    preprocessor = None

# INITIALIZE FASTAPI APP
app = FastAPI(
    title="Flight Delay Prediction API",
    description="API to predict flight delay using a model and preprocessor.",
    version="2.2",
    lifespan=lifespan
)

# DEFINE INPUT/OUTPUT MODELS 
class FlightInput(BaseModel):
    airline: str
    airport: str
    destination_or_origin: str
    aircraft: str
    scheduled_time: str # Expected format: "HH:MM:SS"

class PredictionOutput(BaseModel):
    predicted_delay_minutes: float

# FEATURE ENGINEERING FUNCTION 
def create_features(data: FlightInput) -> pd.DataFrame:
    """Replicates the feature engineering logic from the training pipeline."""
    try:
        flight_time = datetime.strptime(data.scheduled_time, '%H:%M:%S').time()
        scheduled_datetime = datetime.combine(datetime.today(), flight_time)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scheduled_time format. Use HH:MM:SS.")

    df = pd.DataFrame([data.dict()])
    df['scheduled_datetime'] = pd.to_datetime(scheduled_datetime)
    
    dt = df['scheduled_datetime'].iloc[0]
    df['scheduled_day_of_week'] = dt.day_name()
    df['is_weekend'] = df['scheduled_day_of_week'].isin(['Thursday', 'Friday']).astype(int)
    df['Scheduled_Hour_of_Day'] = dt.hour
    df['scheduled_minute'] = dt.minute
    
    hour = dt.hour
    if 5 <= hour <= 11: time_of_day = "Morning"
    elif 12 <= hour <= 16: time_of_day = "Afternoon"
    elif 17 <= hour <= 20: time_of_day = "Evening"
    else: time_of_day = "Night"
    df['scheduled_time_of_day'] = time_of_day
    
    month, day = dt.month, dt.day
    if (month == 3 and day >= 21) or month in [4, 5] or (month == 6 and day <= 20): season = "Spring"
    elif (month == 6 and day >= 21) or month in [7, 8] or (month == 9 and day <= 22): season = "Summer"
    elif (month == 9 and day >= 23) or month in [10, 11] or (month == 12 and day <= 21): season = "Autumn"
    else: season = "Winter"
    df['scheduled_season'] = season
    
    df['is_Nourooz_4'] = 0
    df['is_Nourooz_13'] = 0
    df['Normal_holiday'] = 0
    
    # Return dataframe with columns in the exact order the preprocessor expects
    feature_columns = [
        "airline", "destination_or_origin", "aircraft", "airport",
        "scheduled_day_of_week", "scheduled_season", "scheduled_time_of_day",
        "is_weekend", "is_Nourooz_4", "is_Nourooz_13", "Normal_holiday",
        "Scheduled_Hour_of_Day", "scheduled_minute"
    ]
    return df[feature_columns]

# PREDICTION ENDPOINT 
@app.post("/predict", response_model=PredictionOutput)
def predict(flight_data: FlightInput):
    """
    Predicts flight delay by creating features, transforming them, and then using the model.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not available. Check server logs.")

    # 1. Create the initial features from the raw input
    features_df = create_features(flight_data)
    
    # 2. Transform the features using the loaded preprocessor
    transformed_features = preprocessor.transform(features_df)
    
    # 3. Make the prediction using the transformed data
    prediction = model.predict(transformed_features)
    
    predicted_value = float(prediction[0])
    return {"predicted_delay_minutes": round(predicted_value, 2)}

# HEALTH CHECK ENDPOINT 
@app.get("/health")
def health_check():
    """Checks if the API is running and if the model is loaded."""
    status = "ok" if model is not None and preprocessor is not None else "model_or_preprocessor_not_loaded"
    return {"status": status}