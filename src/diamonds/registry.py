
import os
import pickle
from sklearn.base import BaseEstimator
from diamonds.params import MODEL_FOLDER
 

def save_model(estimator: BaseEstimator, name : str):
    """Save the model to the specified path."""
    # Implement the logic to save the model (e.g., using pickle, joblib, etc.)
    estimator_path = os.path.join(MODEL_FOLDER, f"{name}.pkl")
    with open(estimator_path, "wb") as f:
        pickle.dump(estimator, f)

def load_model(name: str) -> BaseEstimator:
    """Load the model from the specified path."""
    estimator_path = os.path.join(MODEL_FOLDER, f"{name}.pkl")
    with open(estimator_path, "rb") as f:
        estimator = pickle.load(f)
    return estimator