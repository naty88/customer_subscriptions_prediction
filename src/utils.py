import argparse
import logging
import os
from typing import Dict

import numpy as np
import pandas as pd


def create_logging():
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.getLogger("").setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Get predictions with final model.")
    parser.add_argument("test_file_path",
                        type=str,
                        help="Path to file with test data.",
                        default="../data/test_file.xlsx")
    args = parser.parse_args()
    return args


def create_directory(dir_name: str):
    """lay out the directory structure"""
    if not os.path.exists(dir_name):
        logging.info(f"creating {dir_name} directory")
        os.makedirs(dir_name)
    return os.path.abspath(dir_name)


def read_xlsx(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def get_mean_val(scores_dict: Dict[str, np.ndarray], val_name: str):
    if val_name not in scores_dict:
        raise KeyError(f"{val_name} is not found in the scores dictionary.")
    return scores_dict[val_name].mean()
