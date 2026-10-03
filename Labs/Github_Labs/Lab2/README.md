# Using GitHub Actions for Model Training and Versioning

This repository demonstrates how to use GitHub Actions to automate the process of training a machine learning model, storing the model, and versioning it. This allows you to easily update and improve your model in a collaborative environment.

## My modifications to the original lab
- **Dataset:** the original trained on randomly generated synthetic data (a different random size on every run). This version uses the real **Spambase** dataset (UCI, via `fetch_openml('spambase', version=1)`): 4,601 emails x 57 numeric features, binary spam / not-spam label. The labels load as strings, so they are cast to `int` in `src/data.py`.
- **Model:** `RandomForestClassifier` was replaced with `GradientBoostingClassifier` (200 trees, learning rate 0.1, depth 3).
- **Proper evaluation:** training and evaluation share one fixed, stratified 80/20 split (`src/data.py`). The original trained and evaluated on different random data. The model is now evaluated only on the held-out 20%.
- **More metrics:** `evaluate_model.py` now reports Accuracy, Precision, Recall, F1 and ROC AUC (the original only reported F1).
- **Versioned file names:** model files are now named `model_<timestamp>_gb_model.joblib`.
- **Cleaned notebook:** `src/test.ipynb` is now a short walkthrough of the same pipeline instead of a leftover from the RCV1 dataset.

Watch the tutorial video for this lab at [Github action Lab2](https://youtu.be/cj5sXIMZUjQ)


## Prerequisites

- [GitHub](https://github.com) account
- Basic knowledge of Python and machine learning
- Git command-line tool (optional)

## Getting Started

1. **Fork this Repository**: Click the "Fork" button at the top right of this [repository](https://github.com/raminmohammadi/MLOps/) to create your own copy.
3. **Clone Your Repository**:
   ```bash
   git clone https://github.com/your-username/your-forked-repo.git
   cd your-forked-repo

   ```
4. GitHub account
5. Basic knowledge of Python and machine learning
6. Git command-line tool (optional)

# Running the Workflow
## Customize Model Training
1. Modify the `train_model.py` script in the `src/` directory according to your dataset and model requirements. By default it trains a gradient boosting classifier on the Spambase dataset (loaded by `src/data.py`).

## Push Your Changes:
1. Commit your changes and push them to your forked repository.

## GitHub Actions Workflow:
1. Once you push changes to the main branch, the GitHub Actions workflow will be triggered automatically.

## View Workflow Progress:
1. You can track the progress of the workflow by going to the "Actions" tab in your GitHub repository.

## Retrieve the Trained Model:
1. After the workflow completes successfully, the trained model will be stored in the `models/` directory.

# Model Evaluation
The model evaluation is performed automatically within the GitHub Actions workflow. The evaluation results (Accuracy, Precision, Recall, F1 Score and ROC AUC) are stored in the `metrics/` directory.

# Versioning the Model
Each time you run the workflow, a new version of the model is created and stored. You can access and use these models for your projects.

# GitHub Actions Workflow Details
The workflow consists of the following steps:

- Generate and Store Timestamp: A timestamp is generated and stored in a file for versioning.
- Model Training: The `train_model.py` script is executed, which trains a gradient boosting classifier on the training split of the Spambase dataset, logs the run to MLflow (`./mlruns`) and stores the model in the `models/` directory.
- Model Evaluation: The `evaluate_model.py` script is executed to evaluate the model on the held-out test split (Accuracy, Precision, Recall, F1 and ROC AUC), and the results are stored in the `metrics/` directory.
- Store and Version the New Model: The trained model is moved to the `models/` directory with a timestamp-based version.
- Commit and Push Changes: The metrics and updated model are committed to the repository, allowing you to track changes.

# Workflows
Two workflows in `.github/workflows/` run this pipeline (copies live in `workflows/`):

- **`github_lab2_model_calibration_on_push.yml`** runs on every push to `main`. It installs the requirements, runs `train_model.py` and `evaluate_model.py` with a shared timestamp, moves the model to `models/` and the metrics JSON to `metrics/`, and commits both back to the repository.
- **`github_lab2_model_calibration.yml`** runs the same steps on a schedule (daily at 00:00 UTC) so the model is periodically retrained.

Note that, despite the file names, neither workflow applies probability calibration (Platt scaling / isotonic regression); both retrain and re-evaluate the model.

## MLflow tracking
`train_model.py` creates one MLflow experiment per run (`Spambase_<timestamp>`) and logs the parameters, Accuracy and F1 to `./mlruns`. `mlruns/` is git-ignored and GitHub runners are temporary, so tracking data from CI runs is not kept. Only the model and metrics files are committed. Runs made locally stay on your machine (view them with `mlflow ui`).

## Running locally
```bash
pip install -r requirements.txt
timestamp=$(date '+%Y%m%d%H%M%S')
python src/train_model.py --timestamp "$timestamp"
python src/evaluate_model.py --timestamp "$timestamp"   # writes <timestamp>_metrics.json
```
The dataset is downloaded from OpenML on first use. On macOS, if you see `CERTIFICATE_VERIFY_FAILED`, run `export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")` first.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments
- This project uses GitHub Actions for continuous integration and deployment.
- Model training and evaluation are powered by Python and scikit-learn, with experiment tracking by MLflow.
- Dataset: Spambase, UCI Machine Learning Repository.

# Questions or Issues
If you have any questions or encounter issues while using this GitHub Actions workflow, please open an issue in the Issues section of your repository.

