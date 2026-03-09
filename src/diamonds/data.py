import pandas as pd
# Import other necessary libraries here
from pathlib import Path
import seaborn as sns
from src.diamonds.params import DATA_PATH

def load_data(cache = True) -> pd.DataFrame:
    """
    Load the diamonds dataset.

    Parameters
    ----------
    cache : bool, optional
        Whether to cache the dataset, by default True

    Returns
    -------
    pd.DataFrame
        The diamonds dataset
    """
    cache_path = Path(DATA_PATH) / "diamonds.csv"

    if cache and cache_path.exists():
        return pd.read_csv(cache_path)

    df = sns.load_dataset("diamonds")

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The diamonds dataset

    Returns
    -------
    pd.DataFrame
        The cleaned diamonds dataset
    """
    df_clean = df.copy()

    # Remove rows containing at least one zero value
    df_clean = df_clean[(df_clean != 0).all(axis=1)].reset_index(drop=True)

    return df_clean

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned diamonds dataset

    Returns
    -------
    pd.DataFrame
        The preprocessed diamonds dataset
    """
    df_preprocessed = df.copy()

    # Ensure categorical columns are stored as strings
    categorical_cols = df_preprocessed.select_dtypes(include=["category", "object"]).columns
    for col in categorical_cols:
        df_preprocessed[col] = df_preprocessed[col].astype(str)

    return df_preprocessed

def create_X_y(df: pd.DataFrame) ->tuple[pd.DataFrame, pd.Series]:
    """
    Create the feature matrix X and target vector y from the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed diamonds dataset

    Returns
    -------
    (pd.DataFrame, pd.Series)
        The feature matrix X and target vector y
    """
    pass



if __name__ == "__main__":
    df = load_data()
    # df_clean = clean_data(df)
    # df_preprocessed = preprocess_data(df_clean)
    # X, y = create_X_y(df_preprocessed)