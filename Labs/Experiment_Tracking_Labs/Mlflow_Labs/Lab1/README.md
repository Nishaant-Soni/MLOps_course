# MLflow Lab 1: Experiment Tracking, Model Registry and Serving

This lab predicts wine quality (score 3–9) from 12 physicochemical features. It compares three model types: **ElasticNet**, **Random Forest** and **Gradient Boosting**. **MLflow** tracks every step:

1. A **multi-model hyperparameter sweep** runs as nested runs under one parent run.
2. Each run logs **parameters, metrics, plots, the dataset and the model**.
3. The best run, across all model types, is found **by querying MLflow** (`mlflow.search_runs`).
4. The best model is **registered** in the MLflow Model Registry. It gets the `@champion` alias **only if it beats the current champion**.
5. The champion is **loaded by alias** and **served as a REST API** with `mlflow models serve`.

---

## 1. What I changed compared to the original lab

The original Lab 1 (`linear_regression.py`, `starter.ipynb`, `serving.ipynb`, `serving.py`) is **kept unchanged for reference**. My work is in three new scripts: `common.py`, `train.py` and `predict.py`.

### Changes to the MLflow workflow (the core of the lab)

| Area | Original lab | My version | Why it matters |
|---|---|---|---|
| **Tracking backend** | Default `./mlruns` file store. The code skipped model registration on it (`if tracking_url_type_store != "file"`). | **SQLite backend** (`mlflow.db`) with artifacts in `Lab1/mlruns/`, set up in `common.py`. Artifacts always land in `Lab1/` whichever folder you run from. | The Model Registry and aliases work locally without running a server. |
| **Experiment organisation** | Everything went into the `Default` experiment, with unnamed runs. | A named experiment (`wine-quality`), descriptive run names (e.g. `random_forest_n_estimators=200_max_depth=12`), and `run_type` / `model_type` / `dataset` tags. | Runs are easy to find, filter and compare in the UI. |
| **Hyperparameter search** | One run per script call, one model type, with `alpha` / `l1_ratio` passed as positional args. | A **multi-model grid sweep** (20 configs by default: 12 ElasticNet + 4 Random Forest + 4 Gradient Boosting) logged as **nested child runs** under one parent run. `--models` picks which model types to run; `--alphas` / `--l1-ratios` change the ElasticNet grid. | Different model types are compared fairly in one place, on the same data split. |
| **What gets logged** | 2 params, 3 metrics and the model. | Each child run logs `model_type` and its hyperparameters, `val_rmse` / `val_mae` / `val_r2`, a **residuals plot**, a **feature-weights plot** (coefficients for ElasticNet, importances for the tree models), and the model with a **signature and input example**. The parent run logs the **training dataset** (`mlflow.log_input`), the sweep configuration, the **best `val_rmse` per model type**, the best run ID and model type, and the best model's **held-out test metrics**. | Each run is fully reproducible and can be inspected visually. Model types can be compared at a glance on the parent run. |
| **Choosing the best model** | Done by hand in the UI. | `mlflow.search_runs(filter_string="tags.mlflow.parentRunId = '<parent>'", order_by=["metrics.val_rmse ASC"])` picks the winner across all model types. | Model selection is automated and uses the tracking server as the source of truth. |
| **Model Registry** | Registration was skipped on the file store. The original Lab 2 used the deprecated `transition_model_version_stage("Production")`. | Every sweep registers its best model as a new version of `WineQualityModel`, tagged with `val_rmse` and `model_type`. The **`@champion` alias** (the current MLflow API) moves only if the new version has a **lower validation RMSE** than the current champion. This is a **promotion gate**. | A worse retraining run can't silently replace the production model. A better model of a *different type* can replace it. |
| **Loading and serving** | Loaded by hard-coded paths such as `mlruns/0/<run-id>/artifacts/model`. | Loaded and served by alias: `models:/WineQualityModel@champion`. `predict.py` can load the model directly or call the REST server. | Consumers don't need to know run IDs or model types. Promoting a new champion changes what gets served. |
| **MLflow version** | Unpinned, using MLflow 2-era APIs. | Pinned to **MLflow 3.18.0**. Uses MLflow 3 logged-model URIs (`models:/m-...`) for registration, and passes `skops_trusted_types` for tree models (see note below). | Installs the same way every time and uses the current APIs. |

### Changes to the data and models

| Area | Original lab | My version |
|---|---|---|
| **Models** | ElasticNet only. | ElasticNet (inside a `StandardScaler` pipeline), `RandomForestRegressor` and `GradientBoostingRegressor`, all from scikit-learn. |
| **Data** | Red wine only (1,599 rows), downloaded from GitHub on every run. | **Red + white wine combined** (6,497 rows) with an `is_red` indicator, read from local `data/` (no internet needed). The CSVs and the red + white idea come from the original Lab 2 (`Lab2/data`). It's new to Lab 1, but not my own dataset. |
| **Split** | Random 75/25 split. | Seeded **60/20/20 train/validation/test** split. Models are chosen on validation; test is used once, for the final score. |

> **Note on `skops_trusted_types`:** MLflow 3 saves scikit-learn models in the safer `skops` format by default. `skops` refuses to save tree models unless the tree's internal type (`sklearn.tree._tree.Tree`) is explicitly trusted, because a malicious model file could abuse it. We trained these trees ourselves in the same run, so `train.py` trusts that one type, and only for the tree models.

---

## 2. What you need to install

You only need **Python** and **Git**. Everything else is installed by `pip` in step 3.

| Tool | Version | Check with |
|---|---|---|
| Python | **3.12 or newer** (required by `numpy==2.5.4`; tested with 3.12.6) | `python3.12 --version` (macOS/Linux) or `py -3.12 --version` (Windows) |
| Git | any recent version | `git --version` |

**Installing Python 3.12**
- **macOS:** download the installer from <https://www.python.org/downloads/>, or run `brew install python@3.12` if you have Homebrew.
- **Windows:** download the installer from <https://www.python.org/downloads/>. **Tick "Add python.exe to PATH"** during installation.
- **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install python3.12 python3.12-venv`

**Installing Git:** <https://git-scm.com/downloads> (on macOS, running `git` for the first time also offers to install it).

Python packages (installed in step 3 from `requirements.txt`):

```
mlflow==3.18.0
scikit-learn==1.9.1
pandas==3.1.1
numpy==2.5.4
matplotlib==3.11.2
requests==2.34.2
```

---

## 3. Setup

Run these in a terminal (macOS/Linux: Terminal; Windows: PowerShell).

```bash
# 1. Get the code
git clone https://github.com/Nishaant-Soni/MLOps_course.git
cd MLOps_course/Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab1

# 2. Create and activate a virtual environment
python3.12 -m venv .venv            # Windows: py -3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\Activate.ps1
python --version                    # must print Python 3.12.x (or newer)

# 3. Install the dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Your prompt should now start with `(.venv)`. **Run every command below from this `Lab1` folder, with the virtual environment activated.**

> If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

---

## 4. Running the lab

These steps show how the promotion gate behaves over three training runs:

| Step | Command | What happens to `@champion` |
|---|---|---|
| 1 | ElasticNet only | First model ever, so v1 becomes champion |
| 2 | All three model types | Random Forest beats ElasticNet, so v2 becomes champion |
| 3 | A deliberately bad ElasticNet grid | Worse than v2, so the champion **stays** v2 |

MLflow prints some of its own `INFO` lines (and an `mlflow.agent.hint` line) alongside the script output. They're harmless. The outputs below show only what the script prints. All numbers are reproducible because the split and the models use fixed seeds.

### Step 1: Train ElasticNet and register the first champion

```bash
python train.py --models elasticnet
```

This takes about 30 seconds. It creates `mlflow.db` (runs and registry) and `mlruns/` (artifacts) inside `Lab1/`. Expected output:

```
Tracking URI: sqlite:///.../Lab1/mlflow.db
Experiment:   wine-quality
Sweep (12 configs):
  elasticnet         alpha=0.001 l1_ratio=0.1           val_rmse=0.7190
  elasticnet         alpha=0.001 l1_ratio=0.5           val_rmse=0.7189
  elasticnet         alpha=0.001 l1_ratio=0.9           val_rmse=0.7188
  ...
  elasticnet         alpha=0.5 l1_ratio=0.9             val_rmse=0.8536
Best per model type: elasticnet=0.7188
Best run: <run-id> (elasticnet, {'alpha': '0.001', 'l1_ratio': '0.9'}) val_rmse=0.7188 test_rmse=0.7502
Successfully registered model 'WineQualityModel'.
Created version '1' of model 'WineQualityModel'.
Promoted WineQualityModel v1 (elasticnet) to @champion (previous: none)
```

### Step 2: Sweep all three model types (the champion gets replaced)

```bash
python train.py
```

This takes about 1 minute. Expected output:

```
Sweep (20 configs):
  elasticnet         alpha=0.001 l1_ratio=0.1           val_rmse=0.7190
  ...
  random_forest      n_estimators=100 max_depth=8       val_rmse=0.6619
  random_forest      n_estimators=100 max_depth=12      val_rmse=0.6351
  random_forest      n_estimators=200 max_depth=8       val_rmse=0.6597
  random_forest      n_estimators=200 max_depth=12      val_rmse=0.6330
  gradient_boosting  n_estimators=100 learning_rate=0.05 val_rmse=0.6813
  gradient_boosting  n_estimators=100 learning_rate=0.1 val_rmse=0.6729
  gradient_boosting  n_estimators=300 learning_rate=0.05 val_rmse=0.6691
  gradient_boosting  n_estimators=300 learning_rate=0.1 val_rmse=0.6671
Best per model type: elasticnet=0.7188, random_forest=0.6330, gradient_boosting=0.6671
Best run: <run-id> (random_forest, {'n_estimators': '200', 'max_depth': '12'}) val_rmse=0.6330 test_rmse=0.6519
Created version '2' of model 'WineQualityModel'.
Promoted WineQualityModel v2 (random_forest) to @champion (previous: v1 (val_rmse=0.7188))
```

### Step 3: Run a worse sweep (the champion is kept)

```bash
python train.py --models elasticnet --alphas 0.5 1.0 --l1-ratios 0.5
```

Expected ending. Version 3 is registered, but the champion is **not** replaced:

```
Best run: <run-id> (elasticnet, {'alpha': '0.5', 'l1_ratio': '0.5'}) val_rmse=0.8101 test_rmse=0.8346
Created version '3' of model 'WineQualityModel'.
Kept WineQualityModel v2 as @champion: new v3 val_rmse=0.8101 is not better than 0.6330
```

Options for `train.py`: `--models` (any of `elasticnet random_forest gradient_boosting`), `--alphas`, `--l1-ratios`, `--experiment-name`, `--model-name`. Run `python train.py --help` for details.

### Step 4: Explore the results in the MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050
```

Open <http://127.0.0.1:5050> in a browser, then:
- **Experiments → `wine-quality`:** each `model-sweep` parent run expands into its child runs. Select several child runs and click **Compare** to see `val_rmse` by `model_type` and hyperparameters. To show one model type, filter with `params.model_type = "random_forest"`.
- Click a child run → **Artifacts → `plots/`** to see `residuals.png` and `feature_weights.png`. Its logged model includes `MLmodel`, the signature and the input example.
- Click a parent run to see:
  - the **Datasets** section (`wine-quality-train`)
  - the `best_val_rmse_<model_type>` metrics and the `best_test_*` metrics
  - the `best_model_type` / `best_child_run_id` / `promoted` tags
- **Models → `WineQualityModel`:** lists versions 1–3 with their `val_rmse` and `model_type` tags. The `@champion` alias is on version 2.

Press `Ctrl+C` in the terminal to stop the UI.

### Step 5: Load the champion and predict

```bash
python predict.py
```

Expected output (the first 5 rows of the test set, predicted by the Random Forest champion):

```
Predictions from models:/WineQualityModel@champion:
 actual_quality  predicted_quality
              5               5.29
              6               5.51
              6               5.91
              7               6.88
              5               5.23
```

### Step 6: Serve the champion as a REST API

`mlflow models serve` needs to know where the registry is, so set `MLFLOW_TRACKING_URI` first.

**Terminal 1 (start the server):**

```bash
# macOS / Linux
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
# Windows PowerShell
# $env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

mlflow models serve -m "models:/WineQualityModel@champion" --env-manager local -h 127.0.0.1 -p 5001
```

When it's ready, it prints `Uvicorn running on http://127.0.0.1:5001`.

**Terminal 2 (`cd` into `Lab1`, activate the venv again, then send a request):**

```bash
python predict.py --url http://127.0.0.1:5001/invocations
```

Expected output: the same predictions as Step 5, now coming from the REST server:

```
Predictions from http://127.0.0.1:5001/invocations:
 actual_quality  predicted_quality
              5               5.29
              6               5.51
              6               5.91
              7               6.88
              5               5.23
```

You can also call the endpoint with `curl` (macOS/Linux):

```bash
curl -X POST http://127.0.0.1:5001/invocations -H "Content-Type: application/json" -d '{
  "dataframe_split": {
    "columns": ["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide","total_sulfur_dioxide","density","pH","sulphates","alcohol","is_red"],
    "data": [[7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,1.0]]
  }
}'
# -> {"predictions": [5.061831479643485]}
```

Stop the server with `Ctrl+C` in Terminal 1.

### Start over (optional)

Delete the tracking data to reset everything (version numbers start from 1 again):

```bash
rm -rf mlflow.db mlruns          # Windows: Remove-Item -Recurse -Force mlflow.db, mlruns
```

---

## 5. Project structure

```
Lab1/
├── common.py            # NEW: tracking URI, experiment setup, data loading and split (shared)
├── train.py             # NEW: multi-model nested sweep, logging, best-run search, registry + champion gate
├── predict.py           # NEW: predict via models:/WineQualityModel@champion or via the REST server
├── data/
│   ├── winequality-red.csv     
│   └── winequality-white.csv   
├── requirements.txt     # pinned versions for the new scripts
├── .gitignore           # ignores mlflow.db, mlruns/, .venv/
│
├── linear_regression.py / .ipynb   # original lab files, kept for reference only
├── starter.ipynb, serving.ipynb, serving.py
└── tracking-basics.png, autologging-intro.png
```

The original notebooks and `serving.py` are not part of this submission. `serving.py` also needs `xgboost`, which isn't in `requirements.txt`.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `python3.12: command not found` / `py` not found | Python 3.12 isn't installed or isn't on your PATH. See section 2. (Plain `python3` on macOS is often an older version, e.g. 3.9, which can't install these packages.) |
| `ERROR: No matching distribution found for ...` (e.g. `numpy==2.5.4` or `pandas==3.1.1`) during `pip install` | The venv was created with Python older than 3.12. Delete `.venv` and recreate it with `python3.12 -m venv .venv`. |
| `ModuleNotFoundError: No module named 'mlflow'` | The virtual environment isn't active. Run `source .venv/bin/activate` (Windows: `.venv\Scripts\Activate.ps1`). |
| `RESOURCE_DOES_NOT_EXIST: Registered Model with name=WineQualityModel not found` | Run `python train.py` first. For `mlflow models serve`, check that `MLFLOW_TRACKING_URI` is set and that you're in the `Lab1` folder. |
| `Address already in use` | Another process is using the port. Pick a different one (e.g. `-p 5002`) and use it in `--url` too. On macOS, port 5000 is used by AirPlay, which is why this guide uses 5050 and 5001. |
| An `INFO mlflow.agent.hint ...` line at startup | Harmless MLflow 3 message. Hide it with `export MLFLOW_DISABLE_AGENT_HINT=1`. |
| Version numbers or the champion differ from this README | You have runs from earlier attempts. Use "Start over" above, then repeat Steps 1–3. |
