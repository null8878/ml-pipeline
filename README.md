# ML Pipeline

End-to-end machine learning pipeline with model training, evaluation, deployment, and monitoring.

## Features

- Model training pipeline with hyperparameter tuning
- Experiment tracking with MLflow
- Model versioning and registry
- REST API for model serving (FastAPI)
- Data drift detection
- Model performance monitoring
- A/B testing support
- Docker containerization

## Project Structure

```
ml-pipeline/
├── data/              # Sample datasets
├── models/            # Trained model artifacts
├── notebooks/         # Jupyter notebooks for exploration
├── api/               # FastAPI model serving
├── scripts/           # Training and deployment scripts
├── tests/             # Unit and integration tests
└── docker-compose.yml
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python scripts/train.py

# Start API
uvicorn api.main:app --reload --port 8000

# Run tests
pytest tests/ -v
```

## API Endpoints

- `POST /predict` - Get model prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/info` - Model metadata
- `GET /model/metrics` - Performance metrics
- `POST /model/retrain` - Trigger retraining

## License

MIT
