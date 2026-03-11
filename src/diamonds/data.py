from pathlib import Path

import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.model_selection import train_test_split

from diamonds.model import create_preproc
from diamonds.params import DATA_PATH
from diamonds.registry import load_model, save_model


def load_data(cache: bool = True) -> pd.DataFrame:
    """
    Load the diamonds dataset.

    Parameters
    ----------
    cache : bool, default=True
        Whether to use the local cached CSV file if available.

    Returns
    -------
    pd.DataFrame
        The diamonds dataset.
    """
    # Store the raw dataset locally to avoid downloading it every time
    csv_path = Path(DATA_PATH) / "raw" / "diamonds.csv"

    if cache and csv_path.exists():
        logger.info("Loading diamonds dataset from cache...")
        return pd.read_csv(csv_path)

    logger.info("Loading diamonds dataset from seaborn...")
    df = sns.load_dataset("diamonds")

    if cache:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Diamonds dataset cached at {csv_path}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the diamonds dataset by removing rows containing at least
    one zero value in numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        The raw diamonds dataset.

    Returns
    -------
    pd.DataFrame
        The cleaned dataset.
    """
    df_clean = df.copy()
    initial_shape = df_clean.shape

    # Remove rows with invalid zero values in numeric features
    numeric_cols = df_clean.select_dtypes(include="number").columns
    df_clean = df_clean[(df_clean[numeric_cols] != 0).all(axis=1)].reset_index(drop=True)

    logger.info(f"Cleaned dataset: {initial_shape} -> {df_clean.shape}")
    return df_clean


def create_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into features X and target y.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target vector y.
    """
    X = df.drop(columns=["price"])
    y = df["price"]

    logger.info(f"Created X and y: X={X.shape}, y={y.shape}")
    return X, y


def split_X_y(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42,
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into train and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    logger.info(
        f"Split data: X_train={X_train.shape}, X_test={X_test.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}"
    )
    return X_train, X_test, y_train, y_test

def preprocess_data(X: pd.DataFrame, train: bool = True):
    """
    Preprocess the feature matrix using the project's preprocessing pipeline.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix to preprocess.
    train : bool, default=True
        If True, fit a new preprocessor and save it.
        If False, load the saved preprocessor and transform only.

    Returns
    -------
    Transformed feature matrix
        Usually a numpy array or sparse matrix depending on the pipeline.
    """
    if train:
        # Fit only on training data, then save the fitted preprocessor
        logger.info("Fitting preprocessor on training data...")
        preprocessor = create_preproc()
        preprocessor.fit(X)
        save_model(preprocessor, "preprocessor")
    else:
        # Reuse the fitted preprocessor on test/inference data
        logger.info("Loading preprocessor for inference/test data...")
        preprocessor = load_model("preprocessor")

    X_preprocessed = preprocessor.transform(X)
    logger.info(f"Preprocessed data: {X.shape} -> {X_preprocessed.shape}")

    return X_preprocessed


if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)

    X, y = create_X_y(df_clean)
    X_train, X_test, y_train, y_test = split_X_y(X, y)

    # Important: preprocess after the split to avoid data leakage
    X_train_preprocessed = preprocess_data(X_train, train=True)
    X_test_preprocessed = preprocess_data(X_test, train=False)
