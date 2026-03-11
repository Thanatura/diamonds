from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import loguru

from diamonds.registry import save_model, load_model

logger = loguru.logger

# # Set the MLflow 
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_experiment("diamonds")

def create_model(model_name: str) -> BaseEstimator:
    """
    Create an untrained model pipeline including preprocessing and estimator.

    Parameters
    ----------
    model_name : str
        The name of the model (e.g. "ridge", "random_forest")

    Returns
    -------
    BaseEstimator
        A sklearn Pipeline ready to be fitted on raw features.
    """
    estimator = RandomForestRegressor(n_estimators=50, max_depth=10)
    preprocessor = create_preproc()
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("estimator", estimator),
        ]
    )
    logger.info(f"Created the model pipeline: {model_name}")
    return pipeline

def create_preproc() -> Pipeline:
    """
    Create a preprocessing pipeline.
    """
    cat_pipe = Pipeline(
    [ ("cat_imp",SimpleImputer(strategy="most_frequent"))
      ,("ohe",OneHotEncoder(drop="first",sparse_output=False))
        ])
    num_pipe = Pipeline(
    [("knn_imp", KNNImputer(n_neighbors=5))
     ,("scaler", StandardScaler())
      ])
    preprocessor = ColumnTransformer(
    [("numeric",num_pipe, make_column_selector(dtype_include="number"))
    ,("categorical", cat_pipe, make_column_selector(dtype_exclude="number"))
      ]).set_output(transform="pandas")
    return preprocessor

def train_model(pipeline, X_train, y_train , save: bool = True, *args, **kwargs) -> None:
    """Train the pipeline in place and save it."""
    pipeline.fit(X_train, y_train)
    if save: save_model(pipeline, "model", *args, **kwargs)


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    # NB : mae, mse, r2_score, mape
    # Only print the metrics for now
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    logger.info(
        f"Evaluation metrics: MAE={mae:.2f}, MSE={mse:.2f}, R2={r2:.2f}, MAPE={mape:.2%}"
    )
    metrics = {"mae": mae, "mse": mse, "r2": r2, "mape": mape}
    return metrics

