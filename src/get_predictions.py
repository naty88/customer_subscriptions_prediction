import logging
import os
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from modeling_manager.data_transformer import DataTransformer
from utils import read_xlsx, parse_args, create_logging, save_to_csv

logger = logging.getLogger(__name__)


def download_model_pipeline(is_tuned_model_used: bool = False) -> Optional[object]:
    """
    Load the trained model pipeline from the model directory.

    Parameters:
    ----------
    is_tuned_model_used : bool
        Flag to indicate whether to use the tuned model.

    Returns:
    -------
    Optional[object]
        The loaded model pipeline if found, otherwise None.
    """
    model_dir = "../model"
    if not os.path.exists(model_dir):
        logger.error(
            f"There is no model directory '{model_dir}' in the project. Please train a model with '../notebooks/model_selection.ipynb' first."
        )
        sys.exit(0)

    try:
        models_files = os.listdir(model_dir)
        if not models_files:
            logger.error(f"No model files found in the directory '{model_dir}'.")
            return None

        path_to_model = None
        for model_file in models_files:
            if is_tuned_model_used and model_file.endswith("_tuned.pkl"):
                path_to_model = model_file
            if not is_tuned_model_used and not model_file.endswith("_tuned.pkl"):
                path_to_model = model_file
                break

        if not path_to_model:
            if is_tuned_model_used:
                logger.error(
                    f"There is no persistent tuned model in the model directory '{model_dir}'. "
                    "Please perform hyperparameter tuning of the model first with '../notebooks/model_selection.ipynb'."
                )
            else:
                logger.error(f"No appropriate model file found in the directory '{model_dir}'.")
            sys.exit(0)

        logger.info(f"Loading model from {path_to_model}...")
        loaded_pipeline = joblib.load(os.path.join(model_dir, path_to_model))
        logger.info("Model is ready for inference...")
        return loaded_pipeline
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}. Please train the model first.")
        return None


def get_predictions(processed_test_data: pd.DataFrame, is_tuned_model_used: bool):
    """
    Get predictions from the loaded model pipeline.
    """
    loaded_pipeline = download_model_pipeline(is_tuned_model_used)
    if loaded_pipeline:
        y_pred = loaded_pipeline.predict(processed_test_data)
        return y_pred
    return None


def prepare_test_data(path_to_test_data):
    """
    Load and preprocess the test data.
    """
    # load test data and process them first
    test_df = read_xlsx(path_to_test_data)
    # initialize Preprocessor for test data
    data_transformer = DataTransformer(test_df)
    return data_transformer.clean_and_transform()


def show_pos_predictions(y_pred, processed_test_data: pd.DataFrame):
    # get indexes of positive predictions
    positive_pred_idxs = np.where(y_pred == 1)[0]
    if positive_pred_idxs.size > 0:
        df_list = []
        for idx in positive_pred_idxs:
            df = pd.Series(processed_test_data.iloc[idx]).to_frame().T
            df_list.append(df)
        logger.info(f"Following client(s) would subscribe to term deposit:\n{pd.concat(df_list)}")
    else:
        logger.info(f"No client(s) would subscribe to term deposit: there is no positive predictions.")


def produce_predictions(path_to_test_data: str, is_tuned_model_used: bool):
    """
    Display information about positive predictions.
    """
    processed_test_df = prepare_test_data(path_to_test_data)
    y_predicted = get_predictions(processed_test_df, is_tuned_model_used)
    if y_predicted is not None:
        save_to_csv(processed_test_df, y_predicted, is_tuned_model_used)
        show_pos_predictions(y_predicted, processed_test_df)


def main():
    try:
        create_logging()
        args = parse_args()
        test_file_path = args.test_file_path
        is_tuned_model_used = args.tuned
        produce_predictions(test_file_path, is_tuned_model_used)

    except Exception as e:
        raise ValueError(f"Arguments parsing error, check your inputs: {e}")


if __name__ == "__main__":
    main()
