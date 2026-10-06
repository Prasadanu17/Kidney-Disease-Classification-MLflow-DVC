# Kidney Disease Classification — MLflow & DVC

A deep learning project for kidney disease classification using CNN, with **MLflow** for experiment tracking and **DVC** for data/model versioning.

## Workflows

1. Update config.yaml
2. Update secrets.yaml [optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml
10. app.py

## How to Run?

### Steps

### STEP 01 — Clone the Repository

```bash
git clone https://github.com/Prasadanu17/Kidney-Disease-Classification-MLflow-DVC.git
cd Kidney-Disease-Classification-MLflow-DVC
```

### STEP 02 — Create a Conda Environment

After opening the repository, create a new Conda environment:

```bash
conda create -n cnncls python=3.8 -y
```

Activate the environment:

```bash
conda activate cnncls
```

### STEP 03 — Install the Requirements

Install all the required Python packages:

```bash
pip install -r requirements.txt
```

### STEP 04 — Set Up DagsHub for MLflow Tracking

This project uses **[DagsHub](https://dagshub.com)** as a remote MLflow tracking server to log experiments, metrics, and model artifacts.

#### 4.1 — Create a DagsHub Account & Connect the Repository

1. Go to [https://dagshub.com](https://dagshub.com) and sign up / log in.
2. Create a new repository **or** connect your existing GitHub repo:
   - Click **"Connect a repo"** → select your GitHub repository.
3. Once connected, navigate to your repo on DagsHub and click **"Remote"** → **"Experiments"** to get your MLflow tracking URI.

Your tracking URI will look like:

```
https://dagshub.com/<your-username>/Kidney-Disease-Classification-MLflow-DVC.mlflow
```

#### 4.2 — Set Environment Variables

Export your DagsHub credentials as environment variables so MLflow can authenticate:

**Linux / macOS:**
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/anu705545/Kidney-Disease-Classification-MLflow-DVC.mlflow
export MLFLOW_TRACKING_USERNAME=anu705545export MLFLOW_TRACKING_PASSWORD="<set-your-rotated-token-in-this-shell>"
```

**Windows (PowerShell):**
```powershell
$env:MLFLOW_TRACKING_URI = "https://dagshub.com/anu705545/Kidney-Disease-Classification-MLflow-DVC.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "anu705545"
$env:MLFLOW_TRACKING_PASSWORD = "<set-your-rotated-token-in-this-shell>"
```

> **Tip:** You can find your DagsHub access token at **Settings → Access Tokens** on your DagsHub profile.

#### 4.3 — Initialize DagsHub in Code (Alternative)

You can also initialize DagsHub directly in Python instead of setting environment variables:

```python
import dagshub
dagshub.init(
    repo_owner='anu705545',
    repo_name='Kidney-Disease-Classification-MLflow-DVC',
    mlflow=True
)
```

This is already configured in [`research/04_model_evaluation.ipynb`](research/04_model_evaluation.ipynb).

#### 4.4 — View Experiments on DagsHub

After running the pipeline, visit your DagsHub repository and click the **"Experiments"** tab to view all logged runs, metrics, parameters, and artifacts tracked by MLflow.

🔗 **Project on DagsHub:** [https://dagshub.com/anu705545/Kidney-Disease-Classification-MLflow-DVC](https://dagshub.com/anu705545/Kidney-Disease-Classification-MLflow-DVC)

MLFLOW_TRACKING_URI="https://dagshub.com/anu705545/Kidney-Disease-Classification-MLflow-DVC.mlflow"
MLFLOW_TRACKING_USERNAME="anu705545"export MLFLOW_TRACKING_PASSWORD="<set-your-rotated-token-in-this-shell>"
---
Run this to export as env variables:
```bash
 $env:MLFLOW_TRACKING_URI="https://dagshub.com/anu705545/Kidney-Disease-Classification-MLflow-DVC.mlflow"
$env:MLFLOW_TRACKING_USERNAME="anu70554" $env:MLFLOW_TRACKING_PASSWORD = "<set-your-rotated-token-in-this-shell>"
```
---

> Security: a DagsHub access token was previously exposed. Revoke/rotate it immediately; do not save the replacement token in this repository.

> Security: a DagsHub access token was previously exposed. Revoke/rotate it immediately; never save the replacement token in this repository.

### STEP 05 — Configure DVC Remote for DagsHub Storage

Configure the DagsHub DVC remote and local authentication:

```bash
# Add DagsHub DVC remote as default
dvc remote add -d dagshub_remote https://dagshub.com/anu705545/Kidney-Disease-Classification-MLflow-DVC.dvc

# Store authentication credentials securely in local DVC config (.dvc/config.local, ignored by Git)
dvc remote modify --local dagshub_remote auth basic
dvc remote modify --local dagshub_remote user anu705545
dvc remote modify --local dagshub_remote password <your-dagshub-access-token>

# Verify connection to DagsHub remote storage
dvc status -r dagshub_remote
```

> **Security Note:** Authentication credentials are saved exclusively to `.dvc/config.local`, which is git-ignored and never committed to version control.

---

### STEP 06 — Run the Project

Run individual stages or the full pipeline:

```bash
# Run the complete pipeline (Stage 01 to Stage 04)
python main.py

# Or run individual stages:
python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
python src/cnnClassifier/pipeline/stage_02_prepare_base_model.py
python src/cnnClassifier/pipeline/stage_03_model_training.py
python src/cnnClassifier/pipeline/stage_04_model_evaluation.py
```

Reproduce the full DVC pipeline:

```bash
dvc repro
```

Launch the web application:

```bash
python app.py
```

---

## Pipeline Architecture

```
ConfigurationManager (config.yaml, params.yaml)
    │
    ▼
Stage 01: Data Ingestion (Downloads & extracts dataset)
    │
    ▼
Stage 02: Prepare Base Model (VGG16 architecture + custom top layers)
    │
    ▼
Stage 03: Model Training (Trains model with callbacks)
    │
    ▼
Stage 04: Model Evaluation (Inference, metrics, MLflow logging)
    │
    ▼
DVC & DagsHub (Experiment tracking, data & model versioning)
```

---

## Model Evaluation

The Model Evaluation stage assesses the trained CNN classifier on the validation split of kidney CT-scan images (`Normal` vs `Tumor`).

### How Evaluation Works

1. **Model Loading:** Loads the trained weights from `artifacts/training/model.h5` with backward-compatible Keras deserialization.
2. **Validation Dataset:** Builds a non-shuffled validation stream (`validation_split=0.30`) from `artifacts/data_ingestion/kidney-ct-scan-image` (139 validation images across 2 classes).
3. **Inference & Metrics Calculation:**
   - Evaluates Keras loss and categorical accuracy.
   - Computes probability predictions across all batches.
   - Evaluates scikit-learn metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrix.
4. **Local Artifacts:** Persists all metrics to `artifacts/evaluation/scores.json` and `artifacts/evaluation/confusion_matrix.json`.
5. **MLflow Tracking:** Logs parameters, metrics, confusion matrix artifact, and model registry artifacts to the DagsHub MLflow tracking server.

### Evaluation Results

| Metric | Result |
|---|---|
| **Accuracy** | 0.4820 (48.20%) |
| **Precision** | 0.4820 |
| **Recall** | 1.0000 |
| **F1 Score** | 0.6505 |
| **ROC-AUC** | 0.5625 |
| **Loss** | 24.8997 |

#### Confusion Matrix

| | Predicted Normal | Predicted Tumor |
|---|---|---|
| **Actual Normal (72)** | 0 | 72 |
| **Actual Tumor (67)** | 0 | 67 |

> **Analysis:** The metrics reflect a 1-epoch baseline training run (`EPOCHS: 1` in `params.yaml`). The model predicts all images as "Tumor" after 1 epoch of training. Increasing training epochs (e.g., `EPOCHS: 20`) will allow the model to learn proper decision boundaries between Normal and Tumor classes and improve all metrics significantly. The VGG16 backbone is well-suited for this task — just needs more training iterations.

📊 **MLflow Run on DagsHub:** [View Experiment](https://dagshub.com/anu705545/Kidney-Disease-Classification-MLflow-DVC.mlflow/#/experiments/0)

---

## MLflow & DagsHub Integration Overview

| Feature                  | Tool                      | Status |
|--------------------------|---------------------------|--------|
| Experiment Tracking      | MLflow + DagsHub          | ✅ Configured & Active |
| Data Versioning          | DVC                       | ✅ Configured & Active |
| Model Registry           | MLflow (via DagsHub)      | ✅ Configured & Active |
| Remote Storage           | DagsHub Remote (`dagshub_remote`) | ✅ Configured & Authenticated |
| Pipeline Reproducibility | DVC (`dvc.yaml`)          | ✅ Configured & Active |

## Author

- **GitHub:** [Prasadanu17](https://github.com/Prasadanu17)
- **DagsHub:** [anu705545](https://dagshub.com/anu705545)
