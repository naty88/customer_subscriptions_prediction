import logging
import os
import sys
from typing import List

import joblib
import numpy as np
import pandas as pd

from data_transformer import DataTransformer
from utils import read_xlsx, parse_args, create_logging, save_to_csv

logger = logging.getLogger(__name__)


def download_model_pipeline():
    try:
        model_dir = "../model"
        path_to_model = os.listdir(model_dir)[0]
        loaded_pipeline = joblib.load(os.path.join(model_dir, path_to_model))
        logger.info("Model is ready for inference...")
        return loaded_pipeline
    except FileNotFoundError:
        sys.exit(f"There is no file '{path_to_model}'. Please train the model first.")


def get_predictions(processed_test_data: pd.DataFrame):
    loaded_pipeline = download_model_pipeline()
    y_pred = loaded_pipeline.predict(processed_test_data)
    return y_pred


def prepare_test_data(path_to_test_data):
    # load test data and process them first
    test_df = read_xlsx(path_to_test_data)
    # initialize Preprocessor for test data
    data_transformer = DataTransformer(test_df)
    return data_transformer.clean_and_transform()


def show_pos_predictions(y_pred, processed_test_data: pd.DataFrame):
    # get indexes of positive predictions
    positive_pred_idxs = np.where(y_pred == 1)[0]

    df_list = []
    for idx in positive_pred_idxs:
        df = pd.Series(processed_test_data.iloc[idx]).to_frame().T
        df_list.append(df)
    logger.info(f"Following client(s) would subscribe to term deposit:\n{pd.concat(df_list)}")


def produce_predictions(path_to_test_data: str):
    processed_test_df = prepare_test_data(path_to_test_data)
    y_predicted = get_predictions(processed_test_df)
    save_to_csv(processed_test_df, y_predicted)
    show_pos_predictions(y_predicted, processed_test_df)


def main():
    try:
        create_logging()
        args = parse_args()
        test_file_path = args.test_file_path
        produce_predictions(test_file_path)

    except Exception as e:
        raise ValueError(f"Arguments parsing error, check your inputs: {e}")


if __name__ == "__main__":
    main()
