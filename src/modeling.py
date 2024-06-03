import logging
import os
from typing import Dict, Any

import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib

from src.data_transformer import DataTransformer
from src.utils import get_mean_val, create_directory


logger = logging.getLogger(__name__)


def evaluate_models(x_train: pd.DataFrame,
                    y_train: pd.DataFrame,
                    data_transformer: DataTransformer,
                    models: Dict[str, Any]):
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True,
                          random_state=42)  # StratifiedKFold is used by default for classification tasks
    metrics = [
        'accuracy',
        'f1',
        'roc_auc']

    for model_name, model in models.items():
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


def save_best_model(pipeline: ImbPipeline, file_name="trained_model_pipeline.pkl"):
    model_dir = create_directory('../model')
    path_to_pipeline = os.path.join(model_dir, file_name)
    joblib.dump(pipeline, path_to_pipeline)
    logger.info(f"Final model saved to {path_to_pipeline}")


def hyperparameter_tuning(preprocessor, X_train, y_train):
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 500)
        max_depth = trial.suggest_int('max_depth', 10, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor.preprocessor),
            ('smote', SMOTE()),
            ('classifier', model)
        ])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, scoring='f1', cv=skf, n_jobs=-1)
        return cv_scores.mean()

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=50)
    print(f'Best hyperparameters: {study.best_params}')

    best_params = study.best_params
    best_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42
    )

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor.preprocessor),
        ('smote', SMOTE()),
        ('classifier', best_model)
    ])

    pipeline.fit(X_train, y_train)
    save_best_model(pipeline, file_name="trained_model_pipeline_tuned.pkl")
    return pipeline
