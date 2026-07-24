import argparse
import logging
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import mlflow
import mlflow.sklearn
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples=5000, n_features=10):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/training_data.csv', index=False)
    logger.info(f"Generated sample data: {df.shape}")
    return df

def train_model(model_type='random_forest', hyperparameters=None):
    df = pd.read_csv('data/training_data.csv')

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == 'random_forest':
        params = hyperparameters or {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        base_model = RandomForestClassifier(random_state=42)
    elif model_type == 'gradient_boosting':
        params = hyperparameters or {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
        base_model = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Training {model_type} with hyperparameter tuning...")

    grid_search = GridSearchCV(
        base_model,
        params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_

    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)

    y_pred = best_model.predict(X_test_scaled)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    logger.info(f"\nBest Parameters: {grid_search.best_params_}")
    logger.info(f"\nMetrics:\n{classification_report(y_test, y_pred)}")

    with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.log_artifact('data/training_data.csv')

    os.makedirs('models', exist_ok=True)
    model_path = f'models/{model_type}_model.joblib'
    scaler_path = f'models/scaler.joblib'

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Scaler saved to {scaler_path}")

    report = classification_report(y_test, y_pred, output_dict=True)
    return best_model, metrics, report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ML model')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting'])
    parser.add_argument('--generate-data', action='store_true')
    args = parser.parse_args()

    if args.generate_data:
        generate_sample_data()

    if not os.path.exists('data/training_data.csv'):
        logger.info("No training data found. Generating sample data...")
        generate_sample_data()

    model, metrics, report = train_model(args.model_type)

    print("\n=== Training Complete ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"CV Score: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
