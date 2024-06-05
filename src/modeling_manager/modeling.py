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
    """
    Evaluate multiple models using cross-validation and return their performance metrics.

    This function evaluates each model in the provided models list using Stratified K-Fold cross-validation.
    It computes the mean fit time, accuracy, F1 score, and ROC AUC score for each model.

    Parameters
    ----------
    x_train : pd.DataFrame
        The training feature set.
    y_train : pd.DataFrame
        The training labels.
    data_transformer : Any
        An instance of the data transformer to preprocess the data.
    models_list : Dict[str, Any]
        A dictionary mapping model names to their corresponding classes.

    Returns
    -------
    Dict[str, Dict[str, float]]
        A dictionary where keys are model names and values are dictionaries containing mean performance metrics.
    """
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True,
                          random_state=42)  # StratifiedKFold is used by default for classification tasks
    metrics = [
        'accuracy',
        'f1',
        'roc_auc']

    for model_name, model in models_list.items():
        pipeline = build_pipeline(model, data_transformer)

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
    """
    Train the best model on the given training data.
    """
    best_model = models[best_model_name]
    pipeline = build_pipeline(best_model, data_transformer)
    pipeline.fit(x_train, y_train)
    return pipeline


def build_pipeline(model, data_transformer: DataTransformer) -> ImbPipeline:
    """
    Build a machine learning pipeline with preprocessing, SMOTE, and a classifier.
    """
    return ImbPipeline(steps=[
        ('preprocessor', data_transformer.transformer),
        ('smote', SMOTE(sampling_strategy="all")),  # we should balance the data because we have an imbalanced data set
        ('classifier', model)
    ])


def evaluate_model(pipeline: ImbPipeline,
                   x_test: pd.DataFrame,
                   y_test: pd.DataFrame
                   ):
    """
    Evaluate the model pipeline on the test data and log the classification report.
    """
    y_pred = pipeline.predict(x_test)
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    return y_pred


def save_model(pipeline: ImbPipeline, file_name="trained_model_pipeline.pkl"):
    """
    Save the trained model pipeline to a file.
    """
    model_dir = create_directory('../model')
    path_to_pipeline = os.path.join(model_dir, file_name)
    joblib.dump(pipeline, path_to_pipeline)
    logger.info(f"Final model saved to {path_to_pipeline}")
