import argparse
import logging
import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def create_logging():
    """
    Configure logging to output to the console with a specific format.
    """
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.getLogger("").setLevel(logging.INFO)


def parse_args():
    """
    Parse command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Get predictions with final model.")
    parser.add_argument("test_file_path",
                        type=str,
                        help="Path to file with test data.",
                        default="../data/test_file.xlsx")
    parser.add_argument("-t",
                        "--tuned",
                        type=str,
                        help="Use if you want get predictions with tuned model. "
                             "Default: False.",
                        default=False)
    args = parser.parse_args()
    return args


def create_directory(dir_name: str):
    """lay out the directory structure"""
    if not os.path.exists(dir_name):
        logging.info(f"creating {dir_name} directory")
        os.makedirs(dir_name)
    return os.path.abspath(dir_name)


def read_xlsx(path: str) -> pd.DataFrame:
    """
    Read an Excel file into a DataFrame.
    """
    return pd.read_excel(path)


def split_data(df: pd.DataFrame,
               target: str,
               test_size: float = 0.2,
               random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Splits the DataFrame into training and testing sets.
    """
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def get_mean_val(scores_dict: Dict[str, np.ndarray], val_name: str):
    """
    Calculate the mean value of a specific metric from a dictionary of cross-validation scores.
    """
    if val_name not in scores_dict:
        raise KeyError(f"{val_name} is not found in the scores dictionary.")
    return scores_dict[val_name].mean()


def calculate_metrics(y_true: pd.Series,
                      y_pred: pd.Series) -> Dict[str, float]:
    """
    Calculates performance metrics for model evaluation.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy_score": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted")
    }


def save_to_csv(test_df: pd.DataFrame,
                y_pred: pd.Series,
                is_tuned_model_used: bool):
    """
    Save the test DataFrame with predictions to a CSV file.
    """
    test_df.loc[:, "y_pred"] = pd.Series(y_pred, index=test_df.index)
    path_to_file = "../data/test_file_predict_tuned.csv" if is_tuned_model_used else "../data/test_file_predict.csv"
    test_df.to_csv(path_to_file)
    logger.info(f"File with predictions was saved to: {path_to_file}")
