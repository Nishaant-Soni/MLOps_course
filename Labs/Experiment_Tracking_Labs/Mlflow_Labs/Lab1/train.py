"""
Multi-model hyperparameter sweep (ElasticNet, RandomForest, GradientBoosting) on the
combined red + white wine-quality data, tracked with MLflow.

Compared to the original linear_regression.py, this script:
  * logs to a named experiment on a SQLite backend (so the Model Registry works)
  * sweeps several model families as nested child runs under a single parent run
  * logs the training dataset, tags, plots and a model with signature + input example
  * picks the best child run across all model families with mlflow.search_runs
  * registers it and moves the "champion" alias only if it beats the current champion
"""

import argparse
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    CHAMPION_ALIAS,
    EXPERIMENT_NAME,
    MODEL_NAME,
    RANDOM_STATE,
    TARGET,
    load_wine_data,
    setup_experiment,
    setup_tracking,
    split_data,
)

MODEL_TYPES = ["elasticnet", "random_forest", "gradient_boosting"]

# Fixed grids for the tree models; the ElasticNet grid comes from the command line.
TREE_GRIDS = {
    "random_forest": {"n_estimators": [100, 200], "max_depth": [8, 12]},
    "gradient_boosting": {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1]},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODEL_TYPES, default=MODEL_TYPES)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.001, 0.01, 0.1, 0.5])
    parser.add_argument("--l1-ratios", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--model-name", default=MODEL_NAME)
    return parser.parse_args()


def expand_grid(grid):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def build_configs(args):
    """List of (model_type, params) pairs to train in this sweep."""
    grids = {"elasticnet": {"alpha": args.alphas, "l1_ratio": args.l1_ratios}, **TREE_GRIDS}
    return [(model_type, params) for model_type in args.models for params in expand_grid(grids[model_type])]


def build_model(model_type, params):
    if model_type == "elasticnet":
        return make_pipeline(StandardScaler(), ElasticNet(random_state=RANDOM_STATE, **params))
    if model_type == "random_forest":
        return RandomForestRegressor(min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE, **params)
    return GradientBoostingRegressor(max_depth=3, random_state=RANDOM_STATE, **params)


def feature_weights(model):
    """Coefficients for ElasticNet, impurity-based importances for the tree models."""
    estimator = model[-1] if hasattr(model, "steps") else model
    if hasattr(estimator, "coef_"):
        return estimator.coef_, "Coefficient (standardized features)"
    return estimator.feature_importances_, "Feature importance"


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def residuals_figure(actual, pred):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(pred, actual - pred, alpha=0.3, s=10)
    ax.axhline(0, color="red", linewidth=1)
    ax.set_xlabel("Predicted quality")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Validation residuals")
    fig.tight_layout()
    return fig


def feature_weights_figure(feature_names, weights, label, model_type):
    order = np.argsort(np.abs(weights))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(feature_names)[order], weights[order])
    ax.set_xlabel(label)
    ax.set_title(f"{model_type} feature weights")
    fig.tight_layout()
    return fig


def train_child(model_type, params, X_train, y_train, X_val, y_val):
    """Train one configuration inside a nested run and return its validation RMSE."""
    run_name = model_type + "_" + "_".join(f"{k}={v}" for k, v in params.items())
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tags({"run_type": "child", "model_type": model_type})
        model = build_model(model_type, params)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        metrics = eval_metrics(y_val, val_pred)

        mlflow.log_params({"model_type": model_type, **params})
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})

        weights, label = feature_weights(model)
        mlflow.log_figure(residuals_figure(y_val.to_numpy(), val_pred), "plots/residuals.png")
        mlflow.log_figure(
            feature_weights_figure(X_train.columns, weights, label, model_type), "plots/feature_weights.png"
        )
        plt.close("all")

        signature = infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            # MLflow 3 saves sklearn models with skops, which rejects the tree node
            # storage type unless it is explicitly trusted. We trained these trees
            # ourselves in this run, so trusting this one type is safe.
            skops_trusted_types=None if model_type == "elasticnet" else ["sklearn.tree._tree.Tree"],
        )
        # In MLflow 3 models are stored as separate LoggedModel entities, so keep
        # their URI on the run to register/load them later.
        mlflow.set_tag("model_uri", model_info.model_uri)
        param_str = " ".join(f"{k}={v}" for k, v in params.items())
        print(f"  {model_type:<18} {param_str:<34} val_rmse={metrics['rmse']:.4f}")
        return metrics["rmse"]


def current_champion_rmse(client, model_name):
    """Validation RMSE of the model version currently holding the champion alias, if any."""
    try:
        version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except MlflowException:
        return None, None
    run = client.get_run(version.run_id)
    return version, run.data.metrics.get("val_rmse")


def main():
    args = parse_args()
    tracking_uri = setup_tracking()
    setup_experiment(args.experiment_name)
    client = MlflowClient()

    data = load_wine_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    configs = build_configs(args)

    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment:   {args.experiment_name}")

    with mlflow.start_run(run_name="model-sweep") as parent:
        mlflow.set_tags({"run_type": "parent", "dataset": "wine-quality red+white"})
        mlflow.log_params(
            {
                "models": args.models,
                "n_configs": len(configs),
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "test_rows": len(X_test),
            }
        )
        train_df = pd.concat([X_train, y_train.astype("float64")], axis=1)
        mlflow.log_input(
            mlflow.data.from_pandas(train_df, targets=TARGET, name="wine-quality-train"),
            context="training",
        )

        print(f"Sweep ({len(configs)} configs):")
        best_per_type = {}
        for model_type, params in configs:
            rmse = train_child(model_type, params, X_train, y_train, X_val, y_val)
            best_per_type[model_type] = min(rmse, best_per_type.get(model_type, np.inf))
        # One metric per model family so families can be compared on the parent run.
        mlflow.log_metrics({f"best_val_rmse_{t}": v for t, v in best_per_type.items()})

        # Query the tracking server for the best child of *this* sweep, across all model types.
        best = mlflow.search_runs(
            experiment_names=[args.experiment_name],
            filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
            order_by=["metrics.val_rmse ASC"],
            max_results=1,
        ).iloc[0]
        best_rmse = best["metrics.val_rmse"]
        best_model_uri = best["tags.model_uri"]
        best_type = best["params.model_type"]

        # Score the best model once on the held-out test set.
        best_model = mlflow.sklearn.load_model(best_model_uri)
        test_metrics = eval_metrics(y_test, best_model.predict(X_test))
        mlflow.log_metrics({"best_val_rmse": best_rmse, **{f"best_test_{k}": v for k, v in test_metrics.items()}})
        mlflow.set_tags({"best_child_run_id": best.run_id, "best_model_type": best_type})

        best_params = {k: v for k, v in client.get_run(best.run_id).data.params.items() if k != "model_type"}
        print("Best per model type: " + ", ".join(f"{t}={v:.4f}" for t, v in best_per_type.items()))
        print(
            f"Best run: {best.run_id} ({best_type}, {best_params}) "
            f"val_rmse={best_rmse:.4f} test_rmse={test_metrics['rmse']:.4f}"
        )

        # Register the best model and promote it only if it beats the current champion.
        version = mlflow.register_model(best_model_uri, args.model_name)
        client.set_model_version_tag(args.model_name, version.version, "val_rmse", f"{best_rmse:.4f}")
        client.set_model_version_tag(args.model_name, version.version, "model_type", best_type)
        champion, champion_rmse = current_champion_rmse(client, args.model_name)

        if champion is None or champion_rmse is None or best_rmse < champion_rmse:
            client.set_registered_model_alias(args.model_name, CHAMPION_ALIAS, version.version)
            mlflow.set_tag("promoted", "true")
            previous = "none" if champion is None else f"v{champion.version} (val_rmse={champion_rmse:.4f})"
            print(f"Promoted {args.model_name} v{version.version} ({best_type}) to @{CHAMPION_ALIAS} (previous: {previous})")
        else:
            mlflow.set_tag("promoted", "false")
            print(
                f"Kept {args.model_name} v{champion.version} as @{CHAMPION_ALIAS}: "
                f"new v{version.version} val_rmse={best_rmse:.4f} is not better than {champion_rmse:.4f}"
            )


if __name__ == "__main__":
    main()
