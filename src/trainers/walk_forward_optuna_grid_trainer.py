import darts
from darts.models.forecasting.forecasting_model import ForecastingModel
import os
from typing import List, Tuple
import numpy as np
from collections import OrderedDict
from darts.models import LinearRegressionModel
import optuna

from src.utils.model_inputs import ModelInputs
from src.utils.utils import create_combination
from src.utils.backtest_inputs import BacktestInputs

from src.evaluators.walk_forward_evaluator import WalkForwardEvaluator
from src.dataloaders.dill_dataloader import LocalTimeSeriesDillDataloader

"""
This class allows to evaluate an unfitted forecasting model on all locations, using backtesting.
"""


class WalkForwardOptunaGridSearchTrainer:
    features_base = ['pluie', 'etp', 'pompage']
    features_shift = ['pluie_lag_Y-1', 'pluie_lag_Y-2', 'etp_lag_Y-1', 'etp_lag_Y-2']
    features_avg = [
        'pluie_mensuelle', 'pluie_semaine', 'etp_mensuelle', 'etp_semaine', 'pluie_dec_mars',
        'pluie_annuelle_ratio', 'etp_annuelle_ratio']
    combination_shift = create_combination(features_shift)
    combination_avg = create_combination(features_avg)

    def __init__(self,
                 dataloader: LocalTimeSeriesDillDataloader,
                 locations: List[str],
                 backtest_inputs: BacktestInputs,
                 start_dates: List[str],
                 n_trials=100
                 ):
        self.dataloader = dataloader
        self.data = dataloader.load()
        self.locations = locations if locations is not None else list(self.data['targets'].keys())
        self.backtest_inputs = backtest_inputs
        self.start_dates = start_dates
        self.n_trials = n_trials

    def find_best_hyperparameters_all_locations(self):
        return self.find_best_hyperparameters_on_locations(self.locations)

    def find_best_hyperparameters_each_location(self):
        best_model_each_location = []
        best_parameters_each_location = []
        best_features_each_location = []

        for location in self.locations:
            best_model, best_parameters, features = self.find_best_hyperparameters_on_locations([location])
            best_model_each_location.append(best_model)
            best_parameters_each_location.append(best_parameters)
            best_features_each_location.append(features)

        return best_model_each_location, best_parameters_each_location, best_features_each_location

    def find_best_hyperparameters_on_locations(self, locations):
        objective = self.build_objective_reglin_for_locations(locations)
        study = optuna.create_study()
        study.optimize(objective, n_trials=self.n_trials)

        best_model, best_parameters = self.create_model_from_study(study)
        features = self.select_features_from_study(study)

        return best_model, best_parameters, features

    def build_objective_reglin_for_locations(self, locations):
        def objective_reglin(trial):
            features, lags, lags_covs = self.select_lags_and_features_from_trial(trial)
            model = LinearRegressionModel(lags=lags, lags_past_covariates=lags_covs, output_chunk_length=182)
            model_input = ModelInputs(model, features)
            result_per_start_date = []
            for start_date in self.start_dates:
                new_backtest_input = self.backtest_inputs.from_start_date(start_date)
                evaluator = WalkForwardEvaluator(model_input, self.dataloader, new_backtest_input,
                                                 locations=locations)
                result_per_start_date.append(evaluator.evaluate_and_plot_rolling_window_all_locations())
            return np.mean(result_per_start_date)

        return objective_reglin

    def select_features_from_study(self, study) -> List[str]:
        features_lag = study.best_params['features_lag']
        features_avg = study.best_params['features_avg']
        features = self.features_base + features_lag + features_avg

        return features

    def select_lags_and_features_from_trial(self, trial) -> List[str]:
        lags = trial.suggest_int('lag', 5, 100)
        lag_base = trial.suggest_int('lag_base', 5, 100)
        lag_shift = trial.suggest_int('lag_shift', 5, 100)
        lag_avg = trial.suggest_int('lag_avg', 1, 10)
        features_lag = trial.suggest_categorical('features_lag',
                                                 self.combination_shift)
        features_avg = trial.suggest_categorical('features_avg', self.combination_avg)

        features = self.features_base + features_lag + features_avg
        lags_base = [-lag_base for i in range(len(self.features_base))]
        lags_shift = [-lag_shift for i in range(len(self.features_shift))]
        lags_avg = [-lag_avg for i in range(len(features_avg))]
        lags_covs = lags_base + lags_shift + lags_avg

        return features, lags, lags_covs

    def create_model_from_study(self, study) -> Tuple[LinearRegressionModel, OrderedDict]:
        lag = study.best_params['lag']
        lag_base = study.best_params['lag_base']
        lag_lag = study.best_params['lag_shift']
        lag_avg = study.best_params['lag_avg']

        lags_base = [-lag_base for i in range(len(self.features_base))]
        lags_lag = [-lag_lag for i in range(len(self.features_shift))]
        lags_avg = [-lag_avg for i in range(len(self.features_avg))]
        lags_covs = lags_base + lags_lag + lags_avg

        model = LinearRegressionModel(lags=lag, lags_past_covariates=lags_covs, output_chunk_length=182)
        return model, model.model_params
