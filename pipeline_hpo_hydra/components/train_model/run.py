import os
import argparse
import logging

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

import mlflow


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get absolute path of the project's root directory.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def train_model(data_path, model_type, output_path, n_estimators, max_depth):
    mlflow.set_tracking_uri('http://localhost:5000')
    # mlflow.set_experiment('stroke_prediction_v2')

    df = pd.read_csv(os.path.join(ROOT_DIR, data_path))
    X = df.drop(columns=['stroke'])
    y = df['stroke']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

    if model_type == 'logistic_regression':
        logger.info('==================== log reg =================')
        model = LogisticRegression()
    elif model_type == 'random_forest':
        logger.info('==================== RF ===================')
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
    else:
        raise ValueError('available model types: logistic_regression | random_forest')

    with mlflow.start_run():
        mlflow.log_param('model_type', model_type)
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', round(acc, 3))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.to_pickle(model, output_path)
        mlflow.log_artifact(output_path)

        logger.info(f'*** model saved to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=3)

    args = parser.parse_args()
    train_model(
        args.data_path, args.model_type, args.output_path,
        args.n_estimators, args.max_depth
    )
