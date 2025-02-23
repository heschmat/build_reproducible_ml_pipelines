import os
import sys
import logging
import yaml
import pytest

import pandas as pd

import wandb
import mlflow

# Import test functions.
from test_check_data import run_tests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Get absolute path of the project's root directory.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
PATH_CONFIG = os.path.join(ROOT_DIR, 'config.yaml')

with open(PATH_CONFIG, 'r') as f:
    config = yaml.safe_load(f)

def main():
    """Runs data validation checks before training."""
    run = wandb.init(project=config['wandb']['project'], name='check_data', job_type='validation')

    # Load latest dataset from W&B
    artifact = run.use_artifact('iris_dataset:latest', type='dataset')
    artifact_dir = artifact.download()

    # Load dataset
    data_path = os.path.join(artifact_dir, 'iris.csv')
    df = pd.read_csv(data_path)

    logger.info("*** Running data validation tests... ***")

    # Run Pytest checks
    test_results = run_tests(df)

    wandb.log({'data_checks_passed': 'yes' if (test_results == 0) else 'no'})
    wandb.finish()

    mlflow.log_param('stage', 'check_data')
    # MLflow metrics must be numeric.
    mlflow.log_metric('has_checks_passed', 1 if (test_results == 0) else 0)

    if test_results != 0:
        logger.error("❌ Data validation failed! Please check the dataset.")
        sys.exit(1)

    logger.info("✅ All checks passed. Proceeding to training...")

if __name__ == '__main__':
    main()
