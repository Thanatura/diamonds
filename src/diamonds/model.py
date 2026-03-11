import time
from inspect import signature
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.utils.discovery import all_estimators
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from diamonds.registry import save_model

def create_model(model_name: str, random_state: int = 42) -> BaseEstimator:
    """
    Create an untrained regression model from scikit-learn.

    Parameters
    ----------
    model_name : str
        Name of the regression model to create.
        Can be an exact or partial sklearn regressor class name
        (e.g. "Ridge", "RandomForestRegressor", "RandomForest").
    random_state : int, default=42
        Random seed used for models that support it.

    Returns
    -------
    BaseEstimator
        The model ready to be fitted.
    """
    model_name_clean = model_name.lower().replace("_", "").replace(" ", "")

    # Restrict the search to regression estimators only
    estimators = all_estimators(type_filter="regressor")

    exact_matches = []
    partial_matches = []

    for name, estimator_class in estimators:
        normalized_name = name.lower().replace("_", "").replace(" ", "")

        # Prefer an exact normalized match, otherwise allow partial match
        if normalized_name == model_name_clean:
            exact_matches.append((name, estimator_class))
        elif model_name_clean in normalized_name:
            partial_matches.append((name, estimator_class))

    matches = exact_matches if exact_matches else partial_matches

    if not matches:
        raise ValueError(f"No regressor found matching '{model_name}'.")

    if len(matches) > 1:
        available = [name for name, _ in matches]
        raise ValueError(
            f"Ambiguous model name '{model_name}'. Possible matches: {available}"
        )

    selected_name, model_class = matches[0]

    model_params = signature(model_class.__init__).parameters

    # Only pass random_state to models that support it
    if "random_state" in model_params:
        model = model_class(random_state=random_state)
    else:
        model = model_class()

    logger.info(f"Created model: {selected_name}")
    return model


def create_preproc() -> ColumnTransformer:
    """
    Create the preprocessing pipeline.

    Numerical features:
    - KNN imputation
    - Standard scaling

    Categorical features:
    - Most frequent imputation
    - One-hot encoding

    Returns
    -------
    ColumnTransformer
        The preprocessing pipeline.
    """
    cat_pipe = Pipeline(
        steps=[
            ("cat_imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )

    num_pipe = Pipeline(
        steps=[
            ("num_imp", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
        ]
    )

    # Apply different preprocessing steps depending on column type
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", num_pipe, make_column_selector(dtype_include="number")),
            ("categorical", cat_pipe, make_column_selector(dtype_exclude="number")),
        ]
    ).set_output(transform="pandas")

    logger.info("Created preprocessing pipeline")
    return preprocessor


def train_model(model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series, save: bool = True) -> BaseEstimator:
    """
    Train the model on the training data.

    Parameters
    ----------
    model : BaseEstimator
        Model to train.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    save : bool, default=True
        Whether to save the trained model.

    Returns
    -------
    BaseEstimator
        The trained model.
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    logger.info(f"Model trained in {training_time:.3f} seconds")

    if save:
        save_model(model, "model")
        logger.info("Model saved as 'model'")

    return model


def predict(model: BaseEstimator, X: pd.DataFrame) -> pd.Series:
    """
    Make predictions with a trained model.

    Parameters
    ----------
    model : BaseEstimator
        Trained model.
    X : pd.DataFrame
        Input features.

    Returns
    -------
    pd.Series
        Predicted values.
    """
    y_pred = model.predict(X)
    return pd.Series(y_pred, index=X.index, name="prediction")


def evaluate_model(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """
    Evaluate the model on test data.

    Parameters
    ----------
    model : BaseEstimator
        Trained model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True target values.

    Returns
    -------
    dict[str, float]
        Evaluation metrics.
    """
    y_pred = predict(model, X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }

    logger.info(
        "Evaluation metrics | "
        f"MAE={mae:.2f} | "
        f"MSE={mse:.2f} | "
        f"RMSE={rmse:.2f} | "
        f"R2={r2:.4f} | "
        f"MAPE={mape:.2%}"
    )

    return metrics
