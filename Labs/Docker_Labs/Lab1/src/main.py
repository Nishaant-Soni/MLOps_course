# Train and compare several classifiers on the Wine dataset, then save the best one
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_models(random_state):
    # Logistic regression needs scaled features; tree models do not
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state)
        ),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def evaluate(model, X_train, X_test, y_train, y_test):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "cv_f1_macro_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_macro_std": round(float(cv_scores.std()), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "test_f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train classifiers on the Wine dataset.")
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "random_forest", "gradient_boosting"],
                        help="Models to train and compare.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data held out for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed for the split and the models.")
    parser.add_argument("--output-dir", type=Path, default=Path("/app/artifacts"),
                        help="Directory where the model and metrics are written.")
    args = parser.parse_args()

    available = build_models(args.random_state)
    unknown = set(args.models) - set(available)
    if unknown:
        parser.error(f"unknown model(s): {', '.join(sorted(unknown))}; choose from {', '.join(available)}")

    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=args.test_size,
        random_state=args.random_state, stratify=wine.target,
    )
    print(f"Wine dataset: {len(wine.data)} samples, {wine.data.shape[1]} features, "
          f"{len(wine.target_names)} classes")

    results = {}
    trained = {}
    for name in args.models:
        model = available[name]
        results[name] = evaluate(model, X_train, X_test, y_train, y_test)
        trained[name] = model
        r = results[name]
        print(f"  {name:<20} cv_f1={r['cv_f1_macro_mean']:.3f} (+/-{r['cv_f1_macro_std']:.3f})  "
              f"test_acc={r['test_accuracy']:.3f}  test_f1={r['test_f1_macro']:.3f}")

    # Pick the winner on cross-validation so the test set stays an unbiased check
    best_name = max(results, key=lambda n: results[n]["cv_f1_macro_mean"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "wine_model.joblib"
    metrics_path = args.output_dir / "metrics.json"
    joblib.dump(trained[best_name], model_path)
    metrics = {
        "dataset": "sklearn wine",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "class_names": wine.target_names.tolist(),
        "best_model": best_name,
        "results": results,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Best model: {best_name}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
