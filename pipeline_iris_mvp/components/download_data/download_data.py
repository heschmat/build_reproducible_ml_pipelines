import os
import time
import requests
import yaml
import logging

import wandb
import mlflow

# from pipeline_iris_mvp.logger import get_logger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get absolute path of the project's root directory.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
PATH_CONFIG = os.path.join(ROOT_DIR, 'config.yaml')

with open(PATH_CONFIG, 'r') as f:
    config = yaml.safe_load(f)

# Get file absolute path.
DATA_DIR = os.path.join(ROOT_DIR, config['data']['dir'])
FILE_REL_PATH = config['data']['filename']
FILE_ABS_PATH = os.path.join(DATA_DIR, FILE_REL_PATH)

URL = config['data']['url']

def download_data():
    """Downloads the dataset and logs necessary details."""
    start_time = time.time()

    try:
        respone = requests.get(URL, timeout=10)
        # Raise HTTPError for bad responses (4xx and 5xx)
        respone.raise_for_status()

        with open(FILE_ABS_PATH, 'wb') as f:
            f.write(respone.content)

        filesize = os.path.getsize(FILE_ABS_PATH) / 1024  # convert bytes to KB
        duration = time.time() - start_time

        # Log the relative path to file.
        logger.info(f'✅ File downloaded: {FILE_REL_PATH} ({filesize:2f} KB) in {duration:2f} seconds')

        # Log to MLflow
        mlflow.log_param('stage', 'download_data')
        mlflow.log_metric('file_size_kb', filesize)
        mlflow.log_metric('download_duration_sec', duration)

        return filesize, duration

    except requests.exceptions.RequestException as e:
        logger.error(f'❌ Download failed:\n{e}')
        mlflow.log_param('download_failed', True)
        return None, None

def main():
    """Main function to run the component."""
    wandb.init(
        project=config['wandb']['project'],
        name='download_data',
        job_type='download'
    )

    file_size, duration = download_data()

    if file_size is not None:
        # Log the dataset as an artifact in wandb
        artifact = wandb.Artifact('iris_dataset', type='dataset')
        artifact.add_file(FILE_ABS_PATH)
        wandb.log_artifact(artifact_or_path=artifact)

    wandb.finish()

if __name__ == '__main__':
    main()
