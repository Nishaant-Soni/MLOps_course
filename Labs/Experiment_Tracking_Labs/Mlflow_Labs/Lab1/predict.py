"""
Get predictions from the champion model, either by loading it from the
Model Registry (default) or by calling a running `mlflow models serve` endpoint.

    python predict.py                                         # load models:/WineQualityModel@champion
    python predict.py --url http://127.0.0.1:5001/invocations # call the REST server
"""

import argparse

import mlflow
import pandas as pd
import requests

from common import CHAMPION_ALIAS, MODEL_NAME, load_wine_data, setup_tracking, split_data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="REST endpoint of a running model server")
    parser.add_argument("--rows", type=int, default=5, help="number of test rows to score")
    return parser.parse_args()


def main():
    args = parse_args()
    _, _, X_test, _, _, y_test = split_data(load_wine_data())
    X, y = X_test.head(args.rows), y_test.head(args.rows)

    if args.url:
        payload = {"dataframe_split": X.to_dict(orient="split")}
        response = requests.post(args.url, json=payload, timeout=30)
        response.raise_for_status()
        predictions = response.json()["predictions"]
        source = args.url
    else:
        setup_tracking()
        model_uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
        predictions = mlflow.pyfunc.load_model(model_uri).predict(X)
        source = model_uri

    print(f"Predictions from {source}:")
    print(pd.DataFrame({"actual_quality": y.to_numpy(), "predicted_quality": predictions}).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
