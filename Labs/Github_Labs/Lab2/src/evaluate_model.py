import os, json, sys
import joblib
import argparse
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_splits

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    # Access the timestamp
    timestamp = args.timestamp
    try:
        model_version = f'model_{timestamp}_gb_model'  # Use a timestamp as the version
        model = joblib.load(f'{model_version}.joblib')
    except FileNotFoundError:
        raise ValueError('Failed to catching the latest model')

    # Same fixed split as training, so this is held-out data
    _, X_test, _, y_test = load_splits()

    y_predict = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_predict),
        "Precision": precision_score(y_test, y_predict),
        "Recall": recall_score(y_test, y_predict),
        "F1_Score": f1_score(y_test, y_predict),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }

    # Save metrics to a JSON file
    os.makedirs("metrics/", exist_ok=True)
    with open(f'{timestamp}_metrics.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
    print(json.dumps(metrics, indent=4))
