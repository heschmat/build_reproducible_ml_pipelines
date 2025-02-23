import os
import logging
import mlflow.projects
import yaml

import wandb
import mlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get absolute path of the project's root directory.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

PATH_CONFIG = os.path.join(ROOT_DIR, 'config.yaml')

with open(PATH_CONFIG, 'r') as f:
    config = yaml.safe_load(f)

with wandb.init(project=config['wandb']['project'], name='pipeline_run', job_type='pipeline'):
    # Define the pipeline steps.
    components = ['download_data', 'eda', 'check_data', 'train']

    for component in components:
        # component_path = os.path.join(ROOT_DIR, component)
        logger.info(f'*** Started >>> {component}')
        mlflow.projects.run(uri=os.path.join('components', component))

    logger.info('*** ✅ Pipeline execution complete! ***')
