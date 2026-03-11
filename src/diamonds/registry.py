from pathlib import Path
import pickle

import mlflow
import mlflow.sklearn
from loguru import logger
from sklearn.base import BaseEstimator

from diamonds.params import MODEL_PATH, MODEL_REGISTRY


def save_model(estimator: BaseEstimator, name: str) -> str:
    """
    Save a model either locally or to MLflow, depending on MODEL_REGISTRY.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to save.
    name : str
        Model name.

    Returns
    -------
    str
        Local path or MLflow model URI.
    """
    if MODEL_REGISTRY == "local":
        model_path = Path(MODEL_PATH) / f"{name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(estimator, f)

        logger.info(f"Model saved locally at {model_path}")
        return str(model_path)

    if MODEL_REGISTRY == "mlflow":
        if mlflow.active_run() is None:
            raise RuntimeError(
                "No active MLflow run found. Start a run before saving the model."
            )

        mlflow.sklearn.log_model(sk_model=estimator, artifact_path=name)
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{name}"

        logger.info(f"Model saved to MLflow at {model_uri}")
        return model_uri

    raise ValueError(f"Unknown MODEL_REGISTRY value: '{MODEL_REGISTRY}'")


def load_model(name: str) -> BaseEstimator:
    """
    Load a model either locally or from MLflow, depending on MODEL_REGISTRY.

    Parameters
    ----------
    name : str
        Model name (local) or model URI (MLflow).

    Returns
    -------
    BaseEstimator
        The loaded estimator.
    """
    if MODEL_REGISTRY == "local":
        model_path = Path(MODEL_PATH) / f"{name}.pkl"

        with open(model_path, "rb") as f:
            estimator = pickle.load(f)

        logger.info(f"Model loaded locally from {model_path}")
        return estimator

    if MODEL_REGISTRY == "mlflow":
        estimator = mlflow.sklearn.load_model(name)
        logger.info(f"Model loaded from MLflow: {name}")
        return estimator

    raise ValueError(f"Unknown MODEL_REGISTRY value: '{MODEL_REGISTRY}'")
