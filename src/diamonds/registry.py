
from sklearn.base import BaseEstimator
from diamonds.params import MODEL_REGISTRY 

def save_model(model, path):
    """Save the model to the specified path."""
    # Implement the logic to save the model (e.g., using pickle, joblib, etc.)
    pass

def load_model(path) -> BaseEstimator:
    """Load the model from the specified path."""
    # Implement the logic to load the model (e.g., using pickle, joblib, etc.)
    pass