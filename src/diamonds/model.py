from typing import List

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.discovery import all_estimators


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
    estimators: List[tuple[str, BaseEstimator]] = all_estimators()

    candidates = [e for name, e in estimators if model_name.lower() in name.lower()]

    if len(candidates) == 0:
        raise NotImplementedError("no model found for name", model_name)
    return candidates[0]


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


def train_model(model, X_train, y_train):
    pass


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    # NB : mae, mse, r2_score, mape
    # Only print the metrics for now
    pass


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
