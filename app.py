from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <-- ADD THIS IMPORT
from pydantic import BaseModel
import pandas as pd
import joblib

# 1. Initialize the API
app = FastAPI(title="Crime Probability API")

# --- ADD THIS ENTIRE BLOCK ---
# This allows your React app to talk to the API without getting blocked
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (good for local testing)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)
# -----------------------------

# 2. Load the model into memory exactly once when the API starts
try:
    # Make sure this matches the exact name of your newly saved 8-feature model!
    model = joblib.load('crime_predictor_model.joblib')
except Exception as e:
    print(f"Error loading model: {e}")

# 3. Define the strict rules for incoming data (The Bouncer)
class CrimeFeatures(BaseModel):
    # Required fields (The app MUST send these)
    DayOfWeek: int
    Month: int
    is_weekend: int
    lat_bin: float
    lon_bin: float
    
    # Optional fields (If the app omits these, the API defaults to 0.0)
    crimes_last_7_days: float = 0.0
    crimes_last_30_days: float = 0.0
    crimes_last_90_days: float = 0.0

# 4. Create the Prediction Endpoint
@app.post("/predict")
def predict_crime_probability(data: CrimeFeatures):
    try:
        # Convert the incoming JSON payload into a dictionary, then a Pandas DataFrame
        input_data = pd.DataFrame([data.model_dump()])
        
        # Ensure the columns are in the EXACT order the 8-feature model expects
        expected_features = [
            'DayOfWeek', 'Month', 'is_weekend', 
            'crimes_last_7_days', 'crimes_last_30_days', 'crimes_last_90_days', 
            'lat_bin', 'lon_bin'
        ]
        input_data = input_data[expected_features]
        
        # Run the model (predict_proba returns an array, [:, 1] gets the probability of Class 1)
        probability = model.predict_proba(input_data)[:, 1][0]
        
        # Return the percentage to the user
        return {
            "status": "success",
            "probability_of_crime": float(probability),
            "message": f"There is a {probability * 100:.1f}% chance of a crime."
        }
        
    except Exception as e:
        # If anything goes wrong, throw a clean error
        raise HTTPException(status_code=400, detail=str(e))