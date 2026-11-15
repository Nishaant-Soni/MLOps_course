"""Shared settings and data loading for the Lab 1 scripts."""

import os
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "data"

# A SQLite backend (instead of the plain ./mlruns file store) is what enables the
# Model Registry and aliases. MLFLOW_TRACKING_URI can still override it.
DEFAULT_TRACKING_URI = f"sqlite:///{(LAB_DIR / 'mlflow.db').as_posix()}"
ARTIFACT_ROOT = (LAB_DIR / "mlruns").as_uri()

EXPERIMENT_NAME = "wine-quality"
MODEL_NAME = "WineQualityModel"
CHAMPION_ALIAS = "champion"
TARGET = "quality"
RANDOM_STATE = 42


def setup_tracking():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def setup_experiment(name):
    """Create the experiment with artifacts under Lab1/mlruns (independent of the cwd) and make it active."""
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=f"{ARTIFACT_ROOT}/{name}")
    mlflow.set_experiment(name)


def load_wine_data():
    """Combine the red and white wine datasets and add an is_red indicator."""
    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=",")
    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    red["is_red"] = 1
    white["is_red"] = 0
    data = pd.concat([red, white], ignore_index=True)
    data.columns = [c.replace(" ", "_") for c in data.columns]
    # MLflow signatures treat integer columns strictly (no missing values allowed),
    # so cast features to float to keep serving inputs flexible.
    features = data.drop(columns=[TARGET]).astype("float64")
    features[TARGET] = data[TARGET]
    return features


def split_data(data):
    """60/20/20 train/validation/test split with a fixed seed."""
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, train_size=0.6, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem, test_size=0.5, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
