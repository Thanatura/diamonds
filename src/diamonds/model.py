from typing import List

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.discovery import all_estimators
import pandas as pd
import time
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from inspect import signature

import loguru

from diamonds.registry import save_model, load_model

logger = loguru.logger

def create_model(model_name: str, random_state: int = 42) -> BaseEstimator:
    """
    Create an untrained model with the best hyperparameters found during tuning.

    Parameters
    ----------
    model_name : str
        The name of the model (e.g. "ridge", "random_forest")

    Returns
    -------
    BaseEstimator
        The model ready to be fitted
    """
    estimators: List[tuple[str, BaseEstimator]] = all_estimators()

    candidates = [e for name, e in estimators if model_name.lower() in name.lower()]

    if len(candidates) == 0:
        raise NotImplementedError("no model found for name", model_name)
    model_class = candidates[0]

    model_params = signature(model_class).parameters.keys()

    if "random_state" in model_params:
        return model_class(random_state=random_state)
    else:
        return model_class()


def create_preproc() -> Pipeline:
    """
    Create a preprocessing pipeline.
    """

    cat_pipe = Pipeline(
        [
            ("cat_imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", sparse_output=False)),
        ]
    )

    num_pipe = Pipeline(
        [("knn_imp", KNNImputer(n_neighbors=5)), ("scaler", StandardScaler())]
    )

    return ColumnTransformer(
        [
            ("numeric", num_pipe, make_column_selector(dtype_include="number")),
            ("categorical", cat_pipe, make_column_selector(dtype_exclude="number")),
        ]
    ).set_output(transform="pandas")


def train_model(model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series):
    """Train the model on the training data.
    Parameters
    ----------
    model : any
        The model to be trained
    X_train : pd.DataFrame
        The training data
    y_train : pd.Series
        The target values
    Returns
    -------
    None
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")


def predict(model, X):
    """
    Make predictions using the trained model.

    Parameters
    ----------
    model : any
        The trained model
    X : pd.DataFrame
        The raw data

    Returns
    -------
    pd.Series
        The predicted values
    """
    y_pred = model.predict(X)  # Make predictions on the raw data.
    y_pred = pd.Series(
        y_pred, index=X.index
    )  # Convert to a pandas Series with the same index as the input data.
    return y_pred  # Return the predicted values as a pandas Series.


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """Evaluate the model on the test data.
    Parameters
    ----------
    model : any
        The trained model
    X_test : pd.DataFrame
        The test data
    y_test : pd.Series
        The true values

    Returns
    -------
    dict[str, float]
        The evaluation metrics
    """
    # NB : mae, mse, r2_score, mape
    # Only print the metrics for now
    # but return them as a dictionary for later use.
    # y_pred = model.predict(X_test)  # Make predictions on the test data
    y_pred = predict(
        model, X_test
    )  # Use the predict function to make predictions on the test data.
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2score = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("Metrics:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R2 Score: {r2score}")
    print(f"MAPE: {mape}")
    return {
        "MAE": mae,
        "MSE": mse,
        "R2 Score": r2score,
        "MAPE": mape,
    }  # Return the evaluation metrics as a dictionary.
