from diamonds.data import load_data, clean_data, create_X_y
from diamonds.model import create_model, train_model, evaluate_model, save_model
from diamonds.mlflow_utils import mlflow_run, log_metrics, log_params, get_registered_version_for_model_info
from diamonds.registry import update_registered_model_alias

import loguru

logger = loguru.logger

from diamonds.params import MODEL_REGISTRY

def train(
    model_name: str = "baseline",
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
    logger.info(f"Training the model: {model_name}")
    with mlflow_run(run_name="initial_training"):
        log_params(
            {
                "model_name": model_name,
                "test_size": test_size,
                "random_state": random_state,
            }
        )

        df = load_data()
        df_clean = clean_data(df)

        X_train, X_test, y_train, y_test = create_X_y(
            df_clean,
            test_size=test_size,
            random_state=random_state,
        )

        model = create_model(model_name)
        train_model(
            model,
            X_train,
            y_train,
        )

        metrics = evaluate_model(model, X_test, y_test)
        log_metrics(metrics)
        model_info = save_model(model, model_name, X_train, y_train, model_registry=MODEL_REGISTRY)
        return model_info
        


if __name__ == "__main__":
    model_info = train("pipeline",random_state=42)
    version = get_registered_version_for_model_info("diamonds", model_info)
    update_registered_model_alias("diamonds", version, "prod")