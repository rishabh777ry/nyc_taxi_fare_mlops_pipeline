"""
Model training module for NYC Taxi Fare MLOps pipeline.

Trains multiple models (Linear Regression, Random Forest, XGBoost),
performs hyperparameter tuning, logs everything to MLflow, and
registers the best model.
"""

from __future__ import annotations

import logging
import time

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    TrainingConfig,
)

logger = logging.getLogger(__name__)


def setup_mlflow() -> None:
    """Configure MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s, experiment: %s", MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_linear_regression(
    x_train: pd.DataFrame, y_train: pd.Series,
    x_test: pd.DataFrame, y_test: pd.Series,
) -> tuple[LinearRegression, dict[str, float], str]:
    """Train Linear Regression and log to MLflow."""
    with mlflow.start_run(run_name="linear_regression", nested=True) as run:
        logger.info("Training Linear Regression ...")
        start = time.time()

        model = LinearRegression()
        model.fit(x_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(x_test)
        metrics = compute_metrics(y_test, y_pred)

        mlflow.log_params({"model_type": "linear_regression"})
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_time_seconds", train_time)
        mlflow.sklearn.log_model(model, "model")

        logger.info("Linear Regression — RMSE: %.4f, MAE: %.4f, R²: %.4f",
                     metrics["rmse"], metrics["mae"], metrics["r2"])
        return model, metrics, run.info.run_id


def train_random_forest(
    x_train: pd.DataFrame, y_train: pd.Series,
    x_test: pd.DataFrame, y_test: pd.Series,
    config: TrainingConfig | None = None,
) -> tuple[RandomForestRegressor, dict[str, float], str]:
    """Train Random Forest with hyperparameter tuning and log to MLflow."""
    if config is None:
        config = TrainingConfig()

    with mlflow.start_run(run_name="random_forest", nested=True) as run:
        logger.info("Training Random Forest with GridSearchCV ...")
        start = time.time()

        base_model = RandomForestRegressor(random_state=config.random_state, n_jobs=-1)
        grid_search = GridSearchCV(
            base_model,
            config.rf_param_grid,
            cv=config.cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(x_train, y_train)
        train_time = time.time() - start

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(x_test)
        metrics = compute_metrics(y_test, y_pred)

        mlflow.log_params({"model_type": "random_forest", **grid_search.best_params_})
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_time_seconds", train_time)
        mlflow.sklearn.log_model(best_model, "model")

        logger.info("Random Forest — RMSE: %.4f, MAE: %.4f, R²: %.4f (best params: %s)",
                     metrics["rmse"], metrics["mae"], metrics["r2"], grid_search.best_params_)
        return best_model, metrics, run.info.run_id


def train_xgboost(
    x_train: pd.DataFrame, y_train: pd.Series,
    x_test: pd.DataFrame, y_test: pd.Series,
    config: TrainingConfig | None = None,
) -> tuple[XGBRegressor, dict[str, float], str]:
    """Train XGBoost with hyperparameter tuning and log to MLflow."""
    if config is None:
        config = TrainingConfig()

    with mlflow.start_run(run_name="xgboost", nested=True) as run:
        logger.info("Training XGBoost with GridSearchCV ...")
        start = time.time()

        base_model = XGBRegressor(
            random_state=config.random_state,
            objective="reg:squarederror",
            n_jobs=-1,
            verbosity=0,
        )
        grid_search = GridSearchCV(
            base_model,
            config.xgb_param_grid,
            cv=config.cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(x_train, y_train)
        train_time = time.time() - start

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(x_test)
        metrics = compute_metrics(y_test, y_pred)

        mlflow.log_params({"model_type": "xgboost", **grid_search.best_params_})
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_time_seconds", train_time)
        mlflow.xgboost.log_model(best_model, "model")

        logger.info("XGBoost — RMSE: %.4f, MAE: %.4f, R²: %.4f (best params: %s)",
                     metrics["rmse"], metrics["mae"], metrics["r2"], grid_search.best_params_)
        return best_model, metrics, run.info.run_id


def register_best_model(
    results: list[tuple[str, dict[str, float], str]],
) -> tuple[str, str]:
    """
    Select the model with the lowest RMSE and register it in MLflow Model Registry.

    Args:
        results: List of (model_name, metrics_dict, run_id) tuples.

    Returns:
        (best_model_name, model_version) tuple.
    """
    # Sort by RMSE ascending
    results.sort(key=lambda x: x[1]["rmse"])
    best_name, best_metrics, best_run_id = results[0]

    logger.info(
        "Best model: %s (RMSE: %.4f, MAE: %.4f, R²: %.4f)",
        best_name, best_metrics["rmse"], best_metrics["mae"], best_metrics["r2"],
    )

    # Register model
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri, MLFLOW_MODEL_NAME)

    # Transition to Production stage
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=result.version,
        stage="Production",
        archive_existing_versions=True,
    )

    logger.info(
        "Model '%s' v%s promoted to Production stage.",
        MLFLOW_MODEL_NAME, result.version,
    )
    return best_name, result.version


def train_all_models(
    x_train: pd.DataFrame, y_train: pd.Series,
    x_test: pd.DataFrame, y_test: pd.Series,
    config: TrainingConfig | None = None,
) -> tuple[str, str]:
    """
    Train all models, log to MLflow, and register the best one.

    Returns:
        (best_model_name, model_version) tuple.
    """
    if config is None:
        config = TrainingConfig()

    setup_mlflow()

    with mlflow.start_run(run_name="training_pipeline"):
        results = []

        if "linear_regression" in config.models:
            _, metrics, run_id = train_linear_regression(x_train, y_train, x_test, y_test)
            results.append(("linear_regression", metrics, run_id))

        if "random_forest" in config.models:
            _, metrics, run_id = train_random_forest(x_train, y_train, x_test, y_test, config)
            results.append(("random_forest", metrics, run_id))

        if "xgboost" in config.models:
            _, metrics, run_id = train_xgboost(x_train, y_train, x_test, y_test, config)
            results.append(("xgboost", metrics, run_id))

        # Log comparison table
        for name, metrics, _ in results:
            mlflow.log_metrics({f"{name}_rmse": metrics["rmse"], f"{name}_mae": metrics["mae"]})

        # Register best model
        best_name, version = register_best_model(results)

        mlflow.log_param("best_model", best_name)
        mlflow.log_param("model_version", version)

    return best_name, version
