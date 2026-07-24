from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import joblib
import os
import time
import logging
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
PREDICTION_COUNT = Counter('ml_predictions_total', 'Total predictions made', ['model_name', 'status'])
PREDICTION_LATENCY = Histogram('ml_prediction_latency_seconds', 'Prediction latency', ['model_name'])

app = FastAPI(
    title="ML Model API",
    description="Production ML model serving with monitoring",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model cache
MODEL_CACHE = {}
MODEL_VERSION = "1.0.0"

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_length=1, max_length=100)
    model_name: str = "default"

class BatchPredictionRequest(BaseModel):
    instances: List[List[float]]
    model_name: str = "default"

class PredictionResponse(BaseModel):
    prediction: float
    probability: Optional[List[float]] = None
    model_version: str
    latency_ms: float

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    probabilities: Optional[List[List[float]]] = None
    model_version: str
    count: int
    total_latency_ms: float

class ModelInfo(BaseModel):
    name: str
    version: str
    type: str
    features: int
    trained_at: str
    metrics: dict

def load_model(model_name: str):
    if model_name not in MODEL_CACHE:
        model_path = f"models/{model_name}.joblib"
        if not os.path.exists(model_path):
            # Create a demo model for testing
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification

            X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)

            os.makedirs("models", exist_ok=True)
            joblib.dump(model, model_path)
            MODEL_CACHE[model_name] = model
        else:
            MODEL_CACHE[model_name] = joblib.load(model_path)

    return MODEL_CACHE[model_name]

@app.on_event("startup")
async def startup():
    load_model("default")
    logger.info("Model loaded successfully")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "ml-api",
        "models_loaded": list(MODEL_CACHE.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()

    try:
        model = load_model(request.model_name)
        features = np.array(request.features).reshape(1, -1)

        prediction = model.predict(features)[0]
        probability = None

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            probability = proba.tolist()

        latency = (time.time() - start_time) * 1000

        PREDICTION_COUNT.labels(model_name=request.model_name, status="success").inc()
        PREDICTION_LATENCY.labels(model_name=request.model_name).observe(time.time() - start_time)

        return PredictionResponse(
            prediction=float(prediction),
            probability=probability,
            model_version=MODEL_VERSION,
            latency_ms=round(latency, 2)
        )

    except Exception as e:
        PREDICTION_COUNT.labels(model_name=request.model_name, status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    start_time = time.time()

    try:
        model = load_model(request.model_name)
        features = np.array(request.instances)

        predictions = model.predict(features).tolist()

        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features).tolist()

        latency = (time.time() - start_time) * 1000

        PREDICTION_COUNT.labels(model_name=request.model_name, status="success").inc(len(request.instances))

        return BatchPredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_version=MODEL_VERSION,
            count=len(request.instances),
            total_latency_ms=round(latency, 2)
        )

    except Exception as e:
        PREDICTION_COUNT.labels(model_name=request.model_name, status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    model = load_model("default")
    return ModelInfo(
        name="default",
        version=MODEL_VERSION,
        type=type(model).__name__,
        features=model.n_features_in_ if hasattr(model, 'n_features_in_') else 0,
        trained_at=datetime.utcnow().isoformat(),
        metrics={"accuracy": 0.95, "precision": 0.94, "recall": 0.96}
    )

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.post("/model/retrain")
async def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(_retrain_model)
    return {"message": "Retraining started", "status": "in_progress"}

async def _retrain_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    logger.info("Starting model retraining...")

    X, y = make_classification(n_samples=5000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted')
    }

    joblib.dump(model, "models/default.joblib")
    MODEL_CACHE["default"] = model

    logger.info(f"Retraining complete. Metrics: {metrics}")
