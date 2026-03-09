import pandas as pd
import time
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


def create_model(model_name: str) -> BaseEstimator:
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
    pass


def create_preproc() -> Pipeline:
    """
    Create a preprocessing pipeline.
    """
    pass


def train_model(model, X_train, y_train):
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
    model.fit(X_train, y_train)  # Train the model on the training data.
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")


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
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        mean_absolute_percentage_error,
    )

    y_pred = model.predict(X_test)  # Make predictions on the test data.
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2_score = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("Metrics:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R2 Score: {r2_score}")
    print(f"MAPE: {mape}")
    return {
        "MAE": mae,
        "MSE": mse,
        "R2 Score": r2_score,
        "MAPE": mape,
    }  # Return the evaluation metrics as a dictionary.


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
