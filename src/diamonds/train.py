from diamonds.data import clean_data, create_X_y, load_data, preprocess_data, split_X_y
from diamonds.model import (
    create_model,
    create_preproc,
    evaluate_model,
    train_model,
)
from diamonds.registry import save_model
import mlflow
import os

def train(
    model_name: str = "LinearRegression",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Simple end‑to‑end pipeline:

    - load and clean the raw data
    - preprocess it and build X, y
    - split into train / test
    - build the model and preprocessing
    - train, evaluate, and save the trained model
    """
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("Diamonds_Experiment")

    # 1) Data

    df = load_data()
    df_cleaned = clean_data(df)
    X, y = create_X_y(df_cleaned)
    X_train, X_test, y_train, y_test = split_X_y(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2) Model + preprocessing

    estimator = create_model(model_name, random_state=random_state)
    params = {"model_type": model_name, **estimator.get_params()}
    pre_processing = create_preproc()
    pre_processing.fit(X_train)
    X_train_scaled = pre_processing.transform(X_train)
    X_test_scaled = pre_processing.transform(X_test)

    # 3) Evaluation

    train_model(model=estimator, X_train=X_train_scaled, y_train=y_train)
    metrics = evaluate_model(estimator, X_test_scaled, y_test)
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(estimator, 
                                 name=model_name, 
                                 registered_model_name=model_name)

    client = mlflow.MlflowClient()
    model_name = model_name
    model_version_alias = "Best current model"

    # Get the model version using a model URI
    model_uri = f"models:/{model_name}/1"
    model = mlflow.sklearn.load_model(model_uri)

    (model)

if __name__ == "__main__":
    train(model_name="RandomForestRegressor")
