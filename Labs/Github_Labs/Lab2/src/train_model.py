import mlflow, datetime, os, argparse, sys
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import SEED, load_splits


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    # Access the timestamp
    timestamp = args.timestamp
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    X_train, X_test, y_train, y_test = load_splits()

    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Spambase"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    experiment_id = mlflow.create_experiment(f"{experiment_name}")

    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"{dataset_name}"):

        model_params = {"n_estimators": 200, "learning_rate": 0.1,
                        "max_depth": 3, "random_state": SEED}
        mlflow.log_params({"dataset_name": dataset_name,
                           "number of datapoint": X_train.shape[0],
                           "number of dimensions": X_train.shape[1],
                           **model_params})

        model = GradientBoostingClassifier(**model_params)
        model.fit(X_train, y_train)

        y_predict = model.predict(X_test)
        mlflow.log_metrics({'Accuracy': accuracy_score(y_test, y_predict),
                            'F1 Score': f1_score(y_test, y_predict)})

        # Versioned model, moved into models/ by the workflow
        model_filename = f'model_{timestamp}_gb_model.joblib'
        dump(model, model_filename)
