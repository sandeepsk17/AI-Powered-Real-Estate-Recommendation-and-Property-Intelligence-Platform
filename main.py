from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="House Price Prediction API", version="1.0.0")

# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD FILES
# =====================================================

model = joblib.load("house_price_pipeline.pkl")

city_location_mapping = joblib.load("city_location_mapping.pkl")

location_lat_long = joblib.load("location_lat_long.pkl")

# =====================================================
# REQUEST MODEL
# =====================================================


class HouseRequest(BaseModel):
    area: float
    bedrooms: int

    city: str
    location: str

    gymnasium: int = 0
    swimming_pool: int = 0
    clubhouse: int = 0
    power_backup: int = 0
    car_parking: int = 0
    lift_available: int = 0
    ac: int = 0

    resale: int = 0


# =====================================================
# HOME
# =====================================================


@app.get("/")
def home():
    return {"message": "House Price Prediction API Running"}


# =====================================================
# GET CITIES
# =====================================================


@app.get("/cities")
def get_cities():

    return {"cities": sorted(city_location_mapping.keys())}


# =====================================================
# GET LOCATIONS
# =====================================================


@app.get("/locations/{city}")
def get_locations(city: str):

    return {"locations": sorted(city_location_mapping.get(city, []))}


# =====================================================
# PREDICT
# =====================================================


@app.post("/predict")
def predict(data: HouseRequest):

    # -----------------------------------------
    # Get Latitude Longitude from location
    # -----------------------------------------

    if data.location not in location_lat_long:
        raise HTTPException(
            status_code=400, detail=f"Location '{data.location}' not found"
        )

    latitude = location_lat_long[data.location]["Latitude"]
    longitude = location_lat_long[data.location]["Longitude"]

    # -----------------------------------------
    # Create Input DataFrame
    # -----------------------------------------

    input_df = pd.DataFrame(
        [
            {
                "Area": data.area,
                "No. of Bedrooms": data.bedrooms,
                "Latitude": latitude,
                "Longitude": longitude,
                "Resale": data.resale,
                "Gymnasium": data.gymnasium,
                "SwimmingPool": data.swimming_pool,
                "ClubHouse": data.clubhouse,
                "PowerBackup": data.power_backup,
                "CarParking": data.car_parking,
                "LiftAvailable": data.lift_available,
                "AC": data.ac,
                "City": data.city,
                "Location": data.location,
            }
        ]
    )

    # -----------------------------------------
    # Feature Engineering
    # -----------------------------------------

    input_df["Area_per_Bedroom"] = input_df["Area"] / input_df["No. of Bedrooms"]

    input_df["Lat_Long"] = input_df["Latitude"] * input_df["Longitude"]

    input_df["Luxury_Score"] = (
        input_df["Gymnasium"]
        + input_df["SwimmingPool"]
        + input_df["ClubHouse"]
        + input_df["PowerBackup"]
        + input_df["LiftAvailable"]
    )

    # -----------------------------------------
    # Missing Features
    # Training me the but UI me nahi
    # -----------------------------------------

    missing_columns = ["JoggingTrack", "Wardrobe", "Wifi"]

    for col in missing_columns:
        input_df[col] = 0

    # -----------------------------------------
    # Prediction
    # -----------------------------------------

    print(input_df.T)
    
    prediction_log = model.predict(input_df)[0]

    prediction_price = np.expm1(prediction_log)

    return {"predicted_price": round(float(prediction_price), 2)}


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
