import os
import logging

import hydra
from omegaconf import DictConfig

import mlflow.projects


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def main(cfg: DictConfig):

    logger.info(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n{cfg}\n')

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    logger.info('*** running data preparation ... ***')
    mlflow.projects.run(
        uri=os.path.abspath('components/prepare_data'),
        entry_point='main',
        parameters={
            'input_path': cfg.input_path,
            'output_path': cfg.processed_path,
        },
    )

    logger.info('*** training model ... ***')
    mlflow.projects.run(
        uri=os.path.abspath('components/train_model'),
        entry_point='main',
        parameters={
            'data_path': cfg.processed_path,
            'model_type': cfg.model.name,
            'output_path': cfg.output_model,
            'n_estimators': cfg.model.get('n_estimators', 100),
            'max_depth': cfg.model.get('max_depth', 10),
        },
    )

    logger.info('*** pipeline finished *** ')

if __name__ == '__main__':
    main()
