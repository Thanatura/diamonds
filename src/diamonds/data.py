import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split

from diamonds.params import DATA_PATH


def load_data(cache=True) -> pd.DataFrame:
    """
    Load the diamonds dataset.

    Parameters
    ----------
    
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
    numeric_cols = df_clean.select_dtypes(include="number").columns
    df_clean = df_clean[(df_clean[numeric_cols] != 0).all(axis=1)].reset_index(
        drop=True
    )

    return df_clean



def preprocess_data( X: pd.DataFrame
                    , train: bool = True) -> pd.DataFrame:
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
    pass


def create_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
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
    X = df.drop(columns=["price"])
    y = df["price"]

    return X, y


def split_X_y(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    # df_clean = clean_data(df)
    # df_preprocessed = preprocess_data(df_clean)
    # X, y = create_X_y(df_preprocessed)
