from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import uvicorn
import os

app = FastAPI(
    title="Breast Cancer Detection API",
    description="Random Forest model for Malignant/Benign classification",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model  = joblib.load(os.path.join(BASE_DIR, "random_forest_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

FEATURE_NAMES = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
    "mean_smoothness", "mean_compactness", "mean_concavity",
    "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
    "radius_error", "texture_error", "perimeter_error", "area_error",
    "smoothness_error", "compactness_error", "concavity_error",
    "concave_points_error", "symmetry_error", "fractal_dimension_error",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
    "worst_smoothness", "worst_compactness", "worst_concavity",
    "worst_concave_points", "worst_symmetry", "worst_fractal_dimension"
]

class PatientData(BaseModel):
    mean_radius: float = Field(..., example=17.99)
    mean_texture: float = Field(..., example=10.38)
    mean_perimeter: float = Field(..., example=122.8)
    mean_area: float = Field(..., example=1001.0)
    mean_smoothness: float = Field(..., example=0.1184)
    mean_compactness: float = Field(..., example=0.2776)
    mean_concavity: float = Field(..., example=0.3001)
    mean_concave_points: float = Field(..., example=0.1471)
    mean_symmetry: float = Field(..., example=0.2419)
    mean_fractal_dimension: float = Field(..., example=0.07871)
    radius_error: float = Field(..., example=1.095)
    texture_error: float = Field(..., example=0.9053)
    perimeter_error: float = Field(..., example=8.589)
    area_error: float = Field(..., example=153.4)
    smoothness_error: float = Field(..., example=0.006399)
    compactness_error: float = Field(..., example=0.04904)
    concavity_error: float = Field(..., example=0.05373)
    concave_points_error: float = Field(..., example=0.01587)
    symmetry_error: float = Field(..., example=0.03003)
    fractal_dimension_error: float = Field(..., example=0.006193)
    worst_radius: float = Field(..., example=25.38)
    worst_texture: float = Field(..., example=17.33)
    worst_perimeter: float = Field(..., example=184.6)
    worst_area: float = Field(..., example=2019.0)
    worst_smoothness: float = Field(..., example=0.1622)
    worst_compactness: float = Field(..., example=0.6656)
    worst_concavity: float = Field(..., example=0.7119)
    worst_concave_points: float = Field(..., example=0.2654)
    worst_symmetry: float = Field(..., example=0.4601)
    worst_fractal_dimension: float = Field(..., example=0.1189)

class PredictionResponse(BaseModel):
    prediction: str
    prediction_code: int
    malignant_probability: float
    benign_probability: float
    confidence: float
    risk_level: str
    model_used: str

@app.get("/")
def root():
    return {"message": "Breast Cancer Detection API is running!", "model": "Random Forest", "status": "healthy"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "Random Forest", "features": 30, "estimators": model.n_estimators}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    try:
        features = np.array([[getattr(data, f) for f in FEATURE_NAMES]])
        features_scaled = scaler.transform(features)
        prediction = int(model.predict(features_scaled)[0])
        proba = model.predict_proba(features_scaled)[0]

        malignant_prob = float(proba[1])
        benign_prob    = float(proba[0])
        confidence     = float(max(proba))

        if malignant_prob >= 0.70:
            risk_level = "High Risk"
        elif malignant_prob >= 0.40:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        return PredictionResponse(
            prediction="Malignant" if prediction == 1 else "Benign",
            prediction_code=prediction,
            malignant_probability=round(malignant_prob * 100, 2),
            benign_probability=round(benign_prob * 100, 2),
            confidence=round(confidence * 100, 2),
            risk_level=risk_level,
            model_used="Random Forest"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)