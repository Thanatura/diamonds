import os

import mlflow
import mlflow.sklearn
from loguru import logger

from diamonds.data import clean_data, create_X_y, load_data, preprocess_data, split_X_y
from diamonds.model import create_model, evaluate_model, train_model


def train(
    model_name: str = "LinearRegression",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """
    End-to-end training pipeline:
    - load and clean raw data
    - split into train / test
    - preprocess train and test data
    - train the model
    - evaluate the model
    - log params, metrics, and model with MLflow

    Parameters
    ----------
    model_name : str, default="LinearRegression"
        Name of the sklearn regressor to use.
    test_size : float, default=0.2
        Proportion of the dataset used for testing.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict[str, float]
        Evaluation metrics on the test set.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("Diamonds_Experiment")

    with mlflow.start_run():
        logger.info("Loading and preparing data...")
        df = load_data()
        df_cleaned = clean_data(df)

        X, y = create_X_y(df_cleaned)
        X_train, X_test, y_train, y_test = split_X_y(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        # Fit the preprocessor on train only, then reuse it on test
        X_train_preprocessed = preprocess_data(X_train, train=True)
        X_test_preprocessed = preprocess_data(X_test, train=False)

        logger.info(f"Creating model: {model_name}")
        estimator = create_model(model_name, random_state=random_state)

        params = {
            "model_type": model_name,
            "test_size": test_size,
            "random_state": random_state,
            **estimator.get_params(),
        }

        logger.info("Training model...")
        estimator = train_model(
            model=estimator,
            X_train=X_train_preprocessed,
            y_train=y_train,
            save=False,
        )

        logger.info("Evaluating model...")
        metrics = evaluate_model(estimator, X_test_preprocessed, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # In recent MLflow versions, `name` is the preferred argument for the logged model artifact.
        mlflow.sklearn.log_model(
            sk_model=estimator,
            name="model",
            registered_model_name=model_name,
        )

        logger.info(f"Training finished. Metrics: {metrics}")
        return metrics


if __name__ == "__main__":
    train(model_name="RandomForestRegressor")