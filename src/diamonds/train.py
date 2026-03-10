from diamonds.data import (load_data, clean_data
                            , preprocess_data, create_X_y)
from diamonds.model import create_model, train_model, evaluate_model
from diamonds.registry import save_model, load_model


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
    # 1) Data
    df = load_data()
    df_clean = clean_data(df)
    # 2) Model + preprocessing
    X,y = create_X_y(df_clean)
    X_train_preproc = preprocess_data(X_train, train=True)
    X_test_preproc = preprocess_data(X_test, train=False)
    
    model = create_model(model_name)
    train_model(model, X_train_preproc, y_train)
    # 3) Evaluation
    evaluate_model(model, X_test_preproc, y_test)


if __name__ == "__main__":
    train()

