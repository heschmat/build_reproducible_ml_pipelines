import os
import logging
import joblib
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer

import wandb
import mlflow

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get absolute path of the project's root directory.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
PATH_CONFIG = os.path.join(ROOT_DIR, 'config.yaml')

with open(PATH_CONFIG, 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = config['data']['dir']
TRAIN_OUTPUT_DIR = os.path.join(ROOT_DIR, DATA_DIR, 'train')

os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)

def plot_roc_curve(y_test, y_probs, classes):
    """Plots ROC curve and logs it to W&B."""
    plt.figure(figsize=(8, 6))

    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, label=f'Class {class_label}')

    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    roc_path = os.path.join(TRAIN_OUTPUT_DIR, "roc_curve.png")
    plt.savefig(roc_path)
    wandb.log({"roc_curve": wandb.Image(roc_path)})
    plt.close()

def train_model():
    # Initialize W&B run
    run = wandb.init(project=config['wandb']['project'], name='train_run', job_type='train')

    # Use the logged artifact
    artifact = run.use_artifact("iris_dataset:latest", type="dataset")
    artifact_dir = artifact.download()

    # Load dataset
    data_path = os.path.join(artifact_dir, "data.csv")
    data = pd.read_csv(data_path)

    X = data.drop(columns=['species'])
    y = data['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

    # Model training
    model = RandomForestClassifier(n_estimators=10, max_depth=1, random_state=21)
    model.fit(X_train, y_train)

    # Predictions & Evaluation
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)

    # ROC AUC Calculation
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)  # Convert labels to one-hot encoding
    roc_auc = roc_auc_score(y_test_bin, y_probs, multi_class="ovr")

    logger.info(f'Model Accuracy: {acc * 100:.2f}%')
    logger.info(f'Model ROC AUC Score: {roc_auc:.4f}')

    # Log metrics to W&B & MLflow
    wandb.log({"accuracy": acc, "roc_auc": roc_auc})
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", roc_auc)

    # Plot & Log ROC Curve
    plot_roc_curve(y_test_bin, y_probs, lb.classes_)

    # Save Model
    model_path = os.path.join(TRAIN_OUTPUT_DIR, 'model.pkl')
    joblib.dump(model, model_path)
    wandb.save(model_path)

    logger.info('*** Model training complete ***')

def main():
    mlflow.log_param('stage', 'train')
    train_model()

if __name__ == '__main__':
    main()
