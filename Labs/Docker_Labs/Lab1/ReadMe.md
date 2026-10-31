# Docker Lab 1: Wine Classifier Training

This lab packages an ML training job in a Docker image. The container trains
three classifiers on scikit-learn's **Wine** dataset (178 samples, 13 chemical
features, 3 cultivars). It compares them using 5-fold cross-validation and
saves the best model plus a `metrics.json` report to a folder on your machine.
The container stops by itself when training finishes.

## Repository layout

```text
Lab1/
├── Dockerfile            # Image definition: base image, dependencies, user, entrypoint
├── .dockerignore         # Files kept out of the build context
├── .gitignore            # Keeps run outputs (artifacts/) and image tarballs out of git
├── ReadMe.md
├── screenshots/
│   └── run.png           # Output of a successful run
└── src/
    ├── main.py           # Training script: trains, compares and saves the best model
    └── requirements.txt  # Pinned Python packages installed in the image
```

Running the lab creates an `artifacts/` folder holding `wine_model.joblib` and
`metrics.json`. It isn't committed to git.

## Changes from the original lab

| Area | Original | This version | Why it matters |
|------|----------|--------------|----------------|
| Dataset | Iris (4 features) | Wine (13 features, stratified split) | A harder dataset; the stratified split keeps class proportions the same in train and test |
| Model | Single RandomForest | Logistic Regression (scaled), RandomForest and GradientBoosting | Compares model families instead of assuming one is best |
| Model selection | None | Best model picked on 5-fold CV macro-F1 | Choosing on CV keeps the test set an unbiased final check |
| Metrics | None printed | CV F1 (mean/std), test accuracy, test F1 and confusion matrix in `metrics.json` | Results are saved in a machine-readable file, not only printed |
| Configuration | Hard-coded | CLI arguments: `--models`, `--test-size`, `--random-state`, `--output-dir` | One image runs different experiments without rebuilding |
| Output | Saved inside the container and lost on exit | Written to `/app/artifacts`, kept on the host through a bind mount | The trained model survives after the container is removed |
| Base image | `python:3.10` (full) | `python:3.11.18-slim` | Smaller image; the exact patch tag means rebuilds use the same Python |
| Dependencies | Unpinned | Exact versions in `requirements.txt` | Every build installs the same package versions |
| Layer caching | `COPY src/` before `pip install` | `requirements.txt` copied and installed before the code | Editing `main.py` no longer reinstalls every package on rebuild |
| Security | Runs as root | Runs as non-root `appuser` (UID 10001) | Limits the damage if the process in the container is compromised |
| Build context | Everything in the folder | `.dockerignore` excludes `.git`, `artifacts/`, `*.tar`, virtualenvs and caches | Faster builds, and old models or tarballs never get copied into the image |
| Entrypoint | `CMD ["python", "main.py"]` | `ENTRYPOINT` for the script, `CMD` for default arguments | Arguments after the image name go straight to the script |

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Docker Desktop (macOS/Windows) or Docker Engine (Linux) | 23 or newer (tested on 28.0.4) | <https://docs.docker.com/get-started/get-docker/> |
| Git | Any recent version | <https://git-scm.com/downloads> |

- Docker must be running. `docker info` should show a `Server` section.
- The first build needs internet access to download the base image and the
  Python packages. The built image takes about 400 MB of disk space. Later
  builds reuse the cache.
- No local Python install is needed. Everything runs inside the container.

## Get the lab

```sh
git clone https://github.com/Nishaant-Soni/MLOps_course.git
cd MLOps_course/Labs/Docker_Labs/Lab1
```

All of the commands below run from this directory.

## Build

```sh
docker build -t wine-training:1.0 .
```

## Run

Train all three models and save the results to `./artifacts` on your machine.

**macOS / Linux (bash or zsh):**

```sh
mkdir -p artifacts
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  wine-training:1.0
```

**Windows (PowerShell):**

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
docker run --rm -v "${PWD}/artifacts:/app/artifacts" wine-training:1.0
```

What the flags do:

- `--rm` deletes the container after it exits. Training is a one-off job, so
  the container stops by itself and you never need `docker stop`.
- `-v` mounts your `artifacts` folder at `/app/artifacts` inside the container,
  where the script writes its outputs. Without it, the run still works, but the
  outputs are deleted with the container.
- `--user` runs the container as your own user ID. The Dockerfile makes
  `appuser` the default user, and `--user` replaces it for this one run. This
  matters on **Linux**, where `appuser` (UID 10001) can't write to a folder you
  own. Docker Desktop on macOS and Windows handles mount permissions itself, so
  the flag is optional there and not needed in PowerShell.

## Check that it worked

```sh
ls artifacts
```

`ls artifacts` also works in PowerShell, where it prints a table. Either way,
you should see two files:

```text
metrics.json  wine_model.joblib
```

They exist on your machine only if the bind mount worked.

## Expected output

![Container run output](screenshots/run.png)

```text
Wine dataset: 178 samples, 13 features, 3 classes
  logistic_regression  cv_f1=0.993 (+/-0.015)  test_acc=0.972  test_f1=0.971
  random_forest        cv_f1=0.986 (+/-0.027)  test_acc=1.000  test_f1=1.000
  gradient_boosting    cv_f1=0.961 (+/-0.038)  test_acc=0.944  test_f1=0.945
Best model: logistic_regression
Model saved to: /app/artifacts/wine_model.joblib
Metrics saved to: /app/artifacts/metrics.json
```

The model is chosen on cross-validation, not test accuracy. This is why
Logistic Regression wins even though RandomForest scores higher on the
36-sample test set.

## Custom runs

Arguments after the image name replace the defaults. For example, to train
only two models with a larger test split (macOS/Linux):

```sh
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  wine-training:1.0 \
  --models random_forest gradient_boosting --test-size 0.3 --output-dir /app/artifacts
```

On Windows, drop `--user` and use `${PWD}` as in the Run section.

See all options:

```sh
docker run --rm wine-training:1.0 --help
```

## Sample results

From `artifacts/metrics.json` after a default run (`test_size=0.2`,
`random_state=42`):

| Model | CV F1 (mean ± std) | Test accuracy | Test F1 |
|-------|--------------------|---------------|---------|
| **logistic_regression** (best) | 0.9927 ± 0.0147 | 0.9722 | 0.9710 |
| random_forest | 0.9863 ± 0.0274 | 1.0000 | 1.0000 |
| gradient_boosting | 0.9612 ± 0.0378 | 0.9444 | 0.9453 |

Confusion matrices on the test set (rows = true class, columns = predicted):

| | logistic_regression | random_forest | gradient_boosting |
|---|---|---|---|
| class_0 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 |
| class_1 | 0 / 14 / 0 | 0 / 14 / 0 | 1 / 13 / 0 |
| class_2 | 0 / 1 / 9 | 0 / 0 / 10 | 0 / 1 / 9 |

The file looks like this (shortened to one model):

```json
{
  "dataset": "sklearn wine",
  "trained_at": "2026-10-29T18:53:50+00:00",
  "test_size": 0.2,
  "random_state": 42,
  "class_names": ["class_0", "class_1", "class_2"],
  "best_model": "logistic_regression",
  "results": {
    "logistic_regression": {
      "cv_f1_macro_mean": 0.9927,
      "cv_f1_macro_std": 0.0147,
      "test_accuracy": 0.9722,
      "test_f1_macro": 0.971,
      "confusion_matrix": [[12, 0, 0], [0, 14, 0], [0, 1, 9]]
    }
  }
}
```

## Save the image as a tarball (optional)

```sh
docker save -o wine_training.tar wine-training:1.0
```

`-o` writes the file directly, so the same command works in PowerShell.

## Clean up

The container is already gone after each run because of `--rm`. To remove the
image (about 400 MB) and the outputs:

```sh
docker rmi wine-training:1.0
rm -rf artifacts wine_training.tar
```

On Windows PowerShell, use
`Remove-Item -Recurse -Force artifacts, wine_training.tar -ErrorAction SilentlyContinue`
in place of `rm -rf`.
