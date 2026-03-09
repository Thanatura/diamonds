from diamonds.data import clean_data, create_X_y, load_data, preprocess_data, split_X_y
from diamonds.model import (
    create_model,
    create_preproc,
    evaluate_model,
    train_model,
)
from diamonds.registry import save_model


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
    # 1) Data

    df = load_data()
    df_cleaned = clean_data(df)
    df_preprocess = preprocess_data(df_cleaned)
    X, y = create_X_y(df_preprocess)
    X_train, X_test, y_train, y_test = split_X_y(X, y)

    # 2) Model + preprocessing

    estimator = create_model(model_name)
    pre_processing = create_preproc()
    pre_processing.fit(X_train)
    X_train_scaled = pre_processing.transform(X_train)
    X_test_scaled = pre_processing.transform(X_test)

    # 3) Evaluation

    train_model(model=estimator, X_train=X_train_scaled, y_train=y_train)
    evaluate_model(estimator, X_test_scaled, y_test)

    # 4) Persistence

    save_model(estimator, f"models/{model_name}")


if __name__ == "__main__":
    train()
