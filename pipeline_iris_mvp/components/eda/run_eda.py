import os
import yaml
import logging

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import wandb
import mlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get absolute path of the project's root directory.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

PATH_CONFIG = os.path.join(ROOT_DIR, 'config.yaml')

with open(PATH_CONFIG, 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = config['data']['dir']
EDA_OUTPUT_DIR = os.path.join(ROOT_DIR, DATA_DIR, 'eda')
logger.info(f'==> EDA PATH {EDA_OUTPUT_DIR}')

os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

def perform_eda():
    """Perform EDA, save/log plots & necessary metrics."""
    run = wandb.init(project=config['wandb']['project'], name='eda')

    # Use the logged artifact
    artifact = run.use_artifact('iris_dataset:latest', type='dataset')
    artifact_dir = artifact.download()

    data_path = os.path.join(artifact_dir, config['data']['filename'])
    data = pd.read_csv(data_path)

    # Get & log basic summary stats.
    mlflow.log_param('num_samples', len(data))
    mlflow.log_param('num_features', data.shape[1])

    summary_stats = data.describe()
    logger.info(summary_stats)
    wandb.log({'summary_stats': wandb.Table(dataframe=summary_stats)})

    # Histogram
    hist_path = os.path.join(EDA_OUTPUT_DIR, 'sepal_len_hist.png')
    plt.figure(figsize=(8, 6))
    sns.histplot(data['sepal_length'], kde=True)
    plt.title('Sepal Length Distribution')
    plt.savefig(hist_path)
    plt.close()
    wandb.log({'sepal length histogram': wandb.Image(hist_path)})

    # Scatter plot
    scatter_path = os.path.join(EDA_OUTPUT_DIR, 'sepal_len__petal_len.png')
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=data['sepal_length'], y=data['petal_length'])
    plt.title('Sepal Length vs. Petal Length')
    plt.savefig(scatter_path)
    plt.close()
    wandb.log({'sepal length vs. petal length': wandb.Image(scatter_path)})

    logger.info('*** EDA completed successfully')

def main():
    mlflow.log_param('stage', 'eda')
    perform_eda()

if __name__ == '__main__':
    main()
