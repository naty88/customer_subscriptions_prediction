import logging
from typing import Dict, List, Any

import optuna
import pandas as pd
from optuna import Trial
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data_transformer import DataTransformer
from src.modeling import save_best_model

logger = logging.getLogger(__name__)


class Optimizer:
    """
    A class to perform hyperparameter tuning using Optuna.

    Attributes:
    ----------
    model_name : str
        The name of the model to be optimized.
    models_list : Dict[str, Any]
        A dictionary mapping model names to their corresponding classes.
    models_params : Dict[str, Dict[str, List[Any]]]
        A dictionary mapping model names to their hyperparameter options.
    n_trials : int
        The number of trials for Optuna optimization.

    Methods:
    -------
    _suggest_optuna_params(trial: Trial) -> Dict[str, Any]:
        Suggests hyperparameters using Optuna based on the provided trial.
    hyperparameter_tuning(data_transformer: DataTransformer, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        Performs hyperparameter tuning and saves the best model.
    """

    def __init__(
            self,
            model_name: str,
            models_list: Dict[str, Any],
            models_params: Dict[str, Dict[str, List[Any]]],
            n_trials: int = 50
    ):
        self.model_name = model_name
        self.models_list = models_list
        self.models_params = models_params
        self.model_class = self.models_list[self.model_name]
        self.n_trials = n_trials

        logger.info(f"Initialized Optimizer with model: {self.model_name}")

    def _suggest_optuna_params(self, trial: Trial) -> Dict[str, Any]:
        """
        Suggests hyperparameters using Optuna based on the provided trial.

        Parameters:
        ----------
        trial : Trial
            The Optuna trial object.

        Returns:
        -------
        Dict[str, Any]
            A dictionary of suggested hyperparameters.
        """
        if self.model_name not in self.models_params:
            raise ValueError(f"Model '{self.model_name}' not found in models_params dictionary.")

        params = self.models_params[self.model_name]
        optuna_params = {}

        for param_name, param_values in params.items():
            if isinstance(param_values, list):
                if all(isinstance(v, (int, float)) for v in param_values):
                    if all(isinstance(v, int) for v in param_values):
                        # Integer parameters
                        optuna_params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                    else:
                        # Float parameters
                        optuna_params[param_name] = trial.suggest_float(param_name, min(param_values),
                                                                        max(param_values))
                elif all(isinstance(v, str) for v in param_values):
                    # Categorical parameters
                    optuna_params[param_name] = trial.suggest_categorical(param_name, param_values)
                else:
                    raise ValueError(f"Unsupported parameter values for '{param_name}'.")
            else:
                # Fixed value parameters
                optuna_params[param_name] = param_values

        logger.debug(f"Suggested parameters: {optuna_params}")
        return optuna_params

    def hyperparameter_tuning(
            self,
            data_transformer: DataTransformer,
            X_train: pd.DataFrame,
            y_train: pd.Series
    ) -> None:
        """
        Performs hyperparameter tuning and saves the best model.

        Parameters:
        ----------
        data_transformer : DataTransformer
            An instance of the DataTransformer class.
        X_train : pd.DataFrame
            The training feature set.
        y_train : pd.Series
            The training labels.
        """

        def objective(trial: Trial) -> float:
            params = self._suggest_optuna_params(trial)
            model_to_optimize = self.model_class(**params)

            pipeline = ImbPipeline(steps=[
                ('preprocessor', data_transformer.transformer),
                ('smote', SMOTE()),
                ('classifier', model_to_optimize)
            ])

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipeline, X_train, y_train, scoring='f1', cv=skf, n_jobs=-1)
            return cv_scores.mean()

        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=self.n_trials)
        logger.info(f'Best hyperparameters: {study.best_params}')

        best_params = study.best_params
        optimized_model = self.model_class(**best_params)

        pipeline = ImbPipeline(steps=[
            ('preprocessor', data_transformer.transformer),
            ('smote', SMOTE()),
            ('classifier', optimized_model)
        ])

        pipeline.fit(X_train, y_train)
        save_best_model(pipeline, file_name="trained_model_pipeline_tuned.pkl")
        logger.info("Best model saved to trained_model_pipeline_tuned.pkl")
