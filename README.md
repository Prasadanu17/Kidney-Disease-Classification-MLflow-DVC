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
export MLFLOW_TRACKING_USERNAME=anu705545
export MLFLOW_TRACKING_PASSWORD="<set-your-rotated-token-in-this-shell>"
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
MLFLOW_TRACKING_USERNAME="anu705545"
export MLFLOW_TRACKING_PASSWORD="<set-your-rotated-token-in-this-shell>"
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

### DVC Useful Commands

```bash
dvc init          # Initialize DVC in the repository
dvc repro         # Reproduce the full DVC pipeline
dvc dag           # Visualize the pipeline DAG dependency graph
dvc status        # Check status of tracked stages and data
```

---

## About MLflow & DVC

* **MLflow:**
  * Production-grade experiment tracking server
  * Logs hyperparameters, metrics, and models
  * Artifact storage and model registry integration via DagsHub
* **DVC (Data Version Control):**
  * Lightweight dataset and model versioning for reproducible pipelines
  * Pipeline orchestration and dependency caching via `dvc.yaml`
  * Seamless remote data synchronization with DagsHub storage

---

# AWS CI/CD Deployment with GitHub Actions

### 1. Log in to AWS Console
Sign in to your AWS management console at [https://aws.amazon.com](https://aws.amazon.com).

### 2. Create IAM User for Deployment
Create a dedicated IAM user with programmatic access (Access Key ID and Secret Access Key):

**Required Permissions Policies:**
1. `AmazonEC2ContainerRegistryFullAccess` — To push and pull Docker images to/from Amazon ECR.
2. `AmazonEC2FullAccess` — To manage and run workloads on EC2 virtual machines.

### 3. Create Amazon ECR Repository
1. Navigate to **Elastic Container Registry (ECR)**.
2. Create a private repository (e.g., `kidney-app`).
3. Note your repository URI:
   ```text
   556771656148.dkr.ecr.ap-south-1.amazonaws.com/kidney-app
   ```
   * **ECR Login URI:** `556771656148.dkr.ecr.ap-south-1.amazonaws.com`
   * **Repository Name:** `kidney-app`

### 4. Launch Amazon EC2 Instance
1. Launch an **Ubuntu Server** EC2 instance (e.g., `t2.medium` or `t3.medium`).
2. Configure **Security Group** Inbound Rules:
   * **SSH:** Port `22` (Source: Your IP or `0.0.0.0/0`)
   * **Custom TCP:** Port `8080` (Source: `0.0.0.0/0` — for web application traffic)

### 5. Install Docker on EC2
Connect to your EC2 instance via SSH and run:

```bash
# Update packages
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Configure Docker permissions for the ubuntu user
sudo usermod -aG docker ubuntu
newgrp docker
```

### 6. Configure EC2 as GitHub Self-Hosted Runner
1. In your GitHub repository, navigate to:
   **Settings** → **Actions** → **Runners** → **New self-hosted runner**.
2. Select **OS:** `Linux` | **Architecture:** `x64`.
3. Execute the commands provided by GitHub in your EC2 terminal:
   ```bash
   # Download the runner package
   mkdir actions-runner && cd actions-runner
   curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
   tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

   # Configure the runner (follow prompt instructions)
   ./config.sh --url https://github.com/Prasadanu17/Kidney-Disease-Classification-MLflow-DVC --token <RUNNER_TOKEN>

   # Install and run as a system service (runs continuously in background)
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```

### 7. Configure GitHub Repository Secrets
Go to **Settings** → **Secrets and variables** → **Actions** → **New repository secret**, and add:

| Secret Name | Value / Example |
|---|---|
| `AWS_ACCESS_KEY_ID` | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |
| `AWS_REGION` | `ap-south-1` |
| `AWS_ECR_LOGIN_URI` | `556771656148.dkr.ecr.ap-south-1.amazonaws.com` |
| `ECR_REPOSITORY_NAME` | `kidney-app` |

### 8. Automated CI/CD Execution
Once configured, any push to the `main` branch will automatically:
1. Run **Continuous Integration** (linting and checks).
2. Build the Docker container image and push it to **Amazon ECR**.
3. Deploy onto the **EC2 instance** via the self-hosted runner, launching the container on port `8080`.
4. The live application will be accessible at:
   ```text
   http://<EC2-PUBLIC-IP>:8080
   ```