import os
import argparse
import logging

import pandas as pd

from sklearn.preprocessing import LabelEncoder


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get absolute path of the project's root directory.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def prepare_data(input_path, output_path):
    df = pd.read_csv(os.path.join(ROOT_DIR, input_path))
    # logger.info(df.columns)
    # Encode categorical features
    for col in ['gender', 'smoking_status']:
        df[col] = LabelEncoder().fit_transform(df[col])

    df = df[['age', 'gender', 'smoking_status', 'avg_glucose_level', 'stroke']]
    df = df.sample(500, random_state=19)
    os.makedirs(os.path.join(ROOT_DIR, 'data'), exist_ok=True)
    df.to_csv(os.path.join(ROOT_DIR, output_path), index=False)
    logger.info(f'*** data saved to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    prepare_data(args.input_path, args.output_path)
