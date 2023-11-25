import sys

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve

from joblib import dump, load


def load_model(joblib_path: str):
    return load(joblib_path)

# Get test data
def prepare_test_data(joblib_path: str):
    # Load joblib file containing the data preprocessed
    df = load(joblib_path)
    # Drop date column as it's not used anymore
    df.drop(["date"], axis = 1,  inplace=True)
    
    X = df.drop('winner', axis=1)  # X contains only features, i.e all columns except 'winner'
    y = df['winner']  # y contains labels, i.e 'winner' column

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

    return X_test, y_test

# Save confusion matrix and return accuracy score
def eval_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    disp.plot()
    plt.savefig("../img/confusion_matrices/confusion_matrix_mlpc_gridsearch.png")

    return accuracy_score(y_test, predictions)

    

def __main__():
    if len(sys.argv) != 2:
        print("Usage: python eval_clf.py <model_type>")
        print("<model_type> can be 'rf' for Random Forest or 'mlpc' for MLP Classifier")
        sys.exit(1)

    model_type = sys.argv[1]
    X_test, y_test = prepare_test_data("../assets/rera.joblib")

    if model_type == "rf":
        model = load_model("../assets/rf_clf_gridsearch.joblib")
        accuracy = eval_model(model, X_test, y_test)
        print(f"Random Forest model accuracy: {accuracy:<0.3}")
    elif model_type == "mlpc":
        model = load_model("../assets/mlpc_clf_gridsearch.joblib")
        accuracy = eval_model(model, X_test, y_test)
        print(f"MLPC model accuracy: {accuracy:<0.3}")
    else:
        print("Invalid model type. Please choose 'rf' or 'mlpc'.")


if __name__ == "__main__":
    __main__()



