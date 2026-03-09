from sklearn.base import BaseEstimator
from diamonds.params import MODEL_REGISTRY
import pickle


def save_model(model: BaseEstimator, path: str) -> None:
    """Save the model to the specified path."""
    # Implement the logic to save the model (e.g., using pickle, joblib, etc.)
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path: str) -> BaseEstimator:
    """Load the model from the specified path."""
    # Implement the logic to load the model (e.g., using pickle, joblib, etc.)
    with open(path, "rb") as file:
        model = pickle.load(file)
    return model
