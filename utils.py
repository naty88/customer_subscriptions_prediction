from typing import Dict

import numpy as np
import pandas as pd


def read_xlsx_file(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def get_mean_val(scores_dict: Dict[str, np.ndarray], val_name: str):
    if val_name not in scores_dict:
        raise KeyError(f"{val_name} is not found in the scores dictionary.")
    return scores_dict[val_name].mean()
