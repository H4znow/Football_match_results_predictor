import sys

import pandas as pd
import numpy  as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from joblib import load, dump


def prepare_data_for_training(joblib_path: str):
    # Load joblib file containing the data preprocessed
    df = load(joblib_path)
    # Drop date column as it's not used anymore
    df.drop(["date"], axis = 1,  inplace=True)
    
    X = df.drop('winner', axis=1)  # X contains only features, i.e all columns except 'winner'
    y = df['winner']  # y contains labels, i.e 'winner' column

    return X, y


def train_rf_model(X, y):
    # Split data to train only on training data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

    # Pipeline to standardize numerical features
    numeric_features = ["rank_home", "total_points_home", "rank_away", "total_points_away", "home_goals_avg", "away_goals_avg", "home_win_avg", "away_win_avg", "home_last_wins", "away_last_wins"]
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    # Pipeline to encode categorical features
    categorical_features = ["home_team", "away_team", "neutral", "confederation_home", "confederation_away"]
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    # Features preprocessor 
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Final pipeline, including features preprocessing and model training
    # Model's hyperparameters have been found using a Grid search (cf models.ipynb notebook)
    clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=42, max_depth=100, n_estimators=500))]
    )

    # Train the model
    clf.fit(X_train, y_train)

    # Export the model
    dump(clf, "../assets/rf_clf_gridsearch.joblib")


def train_mlpc_model(X, y):
    # Split data to train only on training data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

    # Pipeline to standardize numerical features
    numeric_features = ["rank_home", "total_points_home", "rank_away", "total_points_away", "home_goals_avg", "away_goals_avg", "home_win_avg", "away_win_avg", "home_last_wins", "away_last_wins"]
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    # Pipeline to encode categorical features
    categorical_features = ["home_team", "away_team", "neutral", "confederation_home", "confederation_away"]
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    # Features preprocessor 
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Final pipeline, including features preprocessing and model training
    # Model's hyperparameters have been found using a Grid search (cf models.ipynb notebook)
    clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", MLPClassifier(random_state=42, alpha=0.0001, hidden_layer_sizes=(100,)))]
    )

    # Train the model
    clf.fit(X_train, y_train)

    # Export the model
    dump(clf, "../assets/mlpc_clf_gridsearch.joblib")



def __main__():
    if len(sys.argv) != 2:
        print("Usage: python train_clf.py <model_type>")
        print("<model_type> can be 'rf' for Random Forest or 'mlpc' for MLP Classifier")
        sys.exit(1)

    model_type = sys.argv[1]
    X, y = prepare_data_for_training("../assets/rera.joblib")

    if model_type == "rf":
        train_rf_model(X, y)
        print("Random Forest model trained and saved.")
    elif model_type == "mlpc":
        train_mlpc_model(X, y)
        print("MLP Classifier model trained and saved.")
    else:
        print("Invalid model type. Please choose 'rf' or 'mlpc'.")


if __name__ == "__main__":
    __main__()