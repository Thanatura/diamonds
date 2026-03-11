
import os
import pickle

import loguru
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from diamonds.params import MODEL_FOLDER
from diamonds.mlflow_utils import configure_mlflow

logger = loguru.logger


def save_model(
    pipeline: Pipeline,
    name: str,
    X: pd.DataFrame = None,
    y: pd.Series = None,
    model_registry: str = "local",
) -> None:
    """Save the model to the specified path (local only by default)."""
    estimator_path = os.path.join(MODEL_FOLDER, f"{name}.pkl")
    with open(estimator_path, "wb") as f:
        pickle.dump(pipeline, f)

    if model_registry == "mlflow":
        model_info = log_model_to_mlflow(pipeline, name, X, y)
        return model_info



def log_model_to_mlflow(
    pipeline: Pipeline,
    name: str,
    X: pd.DataFrame = None,
    y: pd.Series = None):
    """
    Log a model to MLflow and create a new model version.

    This does NOT update any aliases.

    Returns
    -------
    Any
        The object returned by ``mlflow.sklearn.log_model`` (e.g. ModelInfo),
        which can be used to identify the logged model.
    """
    configure_mlflow()
    if X is not None:
        X_schema = X.copy()
        int_cols = X_schema.select_dtypes(include=["integer"]).columns.tolist()
        if int_cols:
            X_schema[int_cols] = X_schema[int_cols].astype("float64")
        signature = mlflow.models.infer_signature(X_schema, y)
    else:
        signature = mlflow.models.infer_signature(pipeline)  # or None if you had a fallback
    model_info = mlflow.sklearn.log_model(
        pipeline,
        name=name,
        registered_model_name="diamonds",
        signature=signature,
    )
    logger.info(f"Model logged to MLflow with name: {name}, {model_info=}")
    return model_info



def update_registered_model_alias(
    registered_name: str,
    version_to_update: str,
    alias: str = "prod",
) -> None:
    """
    Update the given alias to point to the latest registered model version.
    """
    configure_mlflow()
    client = MlflowClient()

    client.set_registered_model_alias(
        name=registered_name,
        alias=alias,
        version=version_to_update,
    )
    logger.info(
        f"Updated MLflow alias '{alias}' for model '{registered_name}' to version {version_to_update}"
    )


def load_model(
    name: str,
    model_registry: str = "local",
    model_alias: str = "prod") -> Pipeline:
    """Load the model from the specified path."""
    estimator_path = os.path.join(MODEL_FOLDER, f"{name}.pkl")

    if model_registry == "mlflow":
        configure_mlflow()
        model_uri = f"models:/{name}@{model_alias}"
        estimator = mlflow.sklearn.load_model(model_uri)
        logger.info(
            f"Model loaded from MLflow registry with name: {name} and alias: {model_alias}"
        )
        return estimator

    with open(estimator_path, "rb") as f:
        estimator = pickle.load(f)
    return estimator
