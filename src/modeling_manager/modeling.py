import logging
import os
from typing import Dict, Any

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.modeling_manager.data_transformer import DataTransformer
from src.utils import get_mean_val, create_directory

logger = logging.getLogger(__name__)


def select_model(x_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 data_transformer: DataTransformer,
                 models_list: Dict[str, Any]):
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True,
                          random_state=42)  # StratifiedKFold is used by default for classification tasks
    metrics = [
        'accuracy',
        'f1',
        'roc_auc']

    for model_name, model in models_list.items():
        pipeline = ImbPipeline(steps=[
            ('preprocessor', data_transformer.transformer),
            ('smote', SMOTE()),  # since the data is very imbalanced, it's better to balance them
            ('classifier', model)
        ])

        cv_scores = cross_validate(pipeline, x_train, y_train, cv=skf, scoring=metrics)
        results[model_name] = {'fit_time_mean': get_mean_val(cv_scores, 'fit_time'),
                               'accuracy_mean': get_mean_val(cv_scores, 'test_accuracy'),
                               'f1_mean': get_mean_val(cv_scores, 'test_f1'),
                               'roc_auc_mean': get_mean_val(cv_scores, 'test_roc_auc')}
    return results


def train_best_model(models: Dict[str, Any],
                     best_model_name: str,
                     data_transformer: DataTransformer,
                     x_train: pd.DataFrame,
                     y_train: pd.DataFrame):
    pipeline = ImbPipeline(steps=[
        ('preprocessor', data_transformer.transformer),
        ('smote', SMOTE()),
        ('classifier', models[best_model_name])
    ])

    pipeline.fit(x_train, y_train)
    return pipeline


def evaluate_model(pipeline: ImbPipeline,
                   x_test: pd.DataFrame,
                   y_test: pd.DataFrame
                   ):
    y_pred = pipeline.predict(x_test)
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    return y_pred


def save_model(pipeline: ImbPipeline, file_name="trained_model_pipeline.pkl"):
    model_dir = create_directory('../model')
    path_to_pipeline = os.path.join(model_dir, file_name)
    joblib.dump(pipeline, path_to_pipeline)
    logger.info(f"Final model saved to {path_to_pipeline}")
