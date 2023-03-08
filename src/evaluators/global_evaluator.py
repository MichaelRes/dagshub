import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.dataloaders.dill_dataloader import GlobalTimeSeriesDillDataloader
from src.utils.backtest_inputs import FixedPartitioningInputs
from src.utils.model_inputs import ModelInputs
from src.utils.utils import normalize_series, transform_serie_and_return_scaler, transform_serie

"""
This class allows to evaluate an unfitted forecasting model on all locations, using backtesting.
"""


class GlobalEvaluator:
    aquiferes_6 = ['bioule', 'les_barthes', 'saint_felix', 'saint_porquier', 'saint_porquier', 'verniolles']

    def __init__(self,
                 model_input: ModelInputs,
                 dataloader: GlobalTimeSeriesDillDataloader,
                 valid_6_dataloader: GlobalTimeSeriesDillDataloader,
                 fixed_partitioning_inputs: FixedPartitioningInputs,
                 test_size: int,
                 early_stop: bool,
                 random_state: int = 42,
                 max_samples_per_ts: int = None
                 ):
        self.model = model_input.model
        self.features = model_input.features
        data = dataloader.load()
        self.series = data['series']
        self.covs = [cov[self.features] for cov in data['covs']] if self.features != [] else None
        valid_6_data = valid_6_dataloader.load()
        self.valid_series_6 = valid_6_data['series']
        self.valid_covs_6 = [cov[self.features] for cov in valid_6_data['covs']] if self.features != [] else None
        self.fixed_partitioning_inputs = fixed_partitioning_inputs
        self.metrics = ["MAPE", "sMAPE", "MSE", "MAE", "RMSE", "RMSSE"]
        self.test_size = test_size
        self.random_state = random_state
        self.early_stop = early_stop
        self.max_samples_per_ts = max_samples_per_ts

    def evaluate(self, output_folder: str):
        series_train, series_valid, covs_train, covs_valid = self.train_test_split(self.series, self.covs)
        series_train_before, _ = self.split_on_date(series_train)
        series_valid_before, series_valid_after = self.split_on_date(series_valid)
        covs_train_before, _ = self.split_on_date(covs_train)
        covs_valid_before, _ = self.split_on_date(covs_valid)
        series_valid_before_6, series_valid_after_6 = self.split_on_date(self.valid_series_6)
        covs_valid_before_6, _ = self.split_on_date(self.valid_covs_6)

        print('start fit')
        self.normalize_and_fit(series_train_before, covs_train_before, series_valid_before, covs_valid_before)
        print('fit finished')

        result_metrics = self.evaluate_on_aquiferes(series_valid_before, series_valid_after,
                                                    covs_valid_before,
                                                    output_folder=output_folder)
        result_metrics6 = self.evaluate_on_aquiferes(series_valid_before_6, series_valid_after_6,
                                                     covs_valid_before_6,
                                                     output_folder=output_folder,
                                                     aquiferes_names=self.aquiferes_6)
        self.save_metrics(result_metrics, result_metrics6, output_folder)

    def normalize_and_fit(self, series_train_before, covs_train_before,
                          series_valid_before, covs_valid_before):
        series_train_before_normalized = normalize_series(series_train_before)
        series_valid_before_normalized = normalize_series(series_valid_before)
        covs_train_before_normalized = normalize_series(covs_train_before) if covs_train_before is not None else None
        covs_valid_before_normalized = normalize_series(covs_valid_before) if covs_valid_before is not None else None
        print(f'max_samples_per_ts: {self.max_samples_per_ts}')
        if self.early_stop:
            self.model.fit(series=series_train_before_normalized,
                           val_series=series_valid_before_normalized,
                           max_samples_per_ts=self.max_samples_per_ts,
                           past_covariates=covs_train_before_normalized,
                           val_past_covariates=covs_valid_before_normalized)
        else:
            self.model.fit(series=series_train_before_normalized,
                           max_samples_per_ts=self.max_samples_per_ts,
                           past_covariates=covs_train_before_normalized)

    def evaluate_on_aquiferes(self, series_before, series_after,
                              covs_before,
                              output_folder=None, aquiferes_names=None):
        metrics = np.zeros((len(series_before), len(self.metrics)))

        for i in range(len(series_before)):
            if covs_before is None:
                preds = self.predict(serie=series_before[i],
                                     covs=None)
            else:
                preds = self.predict(serie=series_before[i],
                                     covs=covs_before[i])

            aquifere_name = i if aquiferes_names is None else aquiferes_names[i]
            metrics[i, :] = self.fixed_partitioning_inputs.metric(series_after[i]
                                                                  [:self.fixed_partitioning_inputs.forecast_horizon],
                                                                  preds, series_before[i])

            if output_folder is not None:
                self.save_plots(series_before[i],
                                series_after[i][:self.fixed_partitioning_inputs.forecast_horizon],
                                preds,
                                aquifere_name,
                                metrics[i, :],
                                output_folder)

        return metrics

    def predict(self, serie, covs):
        serie_normalized, scaler = transform_serie_and_return_scaler(serie)
        if covs is None:
            past_covariates = None
        else:
            past_covariates = [transform_serie(covs)]

        preds_normalized = self.model.predict(n=self.fixed_partitioning_inputs.forecast_horizon,
                                              series=[serie_normalized],
                                              past_covariates=past_covariates)[0]
        return scaler.inverse_transform(preds_normalized)

    def train_test_split(self, series, covs):
        if covs is not None:
            series_train, series_valid, covs_train, covs_valid = train_test_split(series, covs, test_size=self.test_size,
                                                                                  random_state=self.random_state)
            return series_train, series_valid, covs_train, covs_valid

        else:
            series_train, series_valid = train_test_split(series, test_size=self.test_size,
                                                          random_state=self.random_state)
            return series_train, series_valid, None, None

    def split_on_date(self, series):
        if series is None:
            return None, None
        else:
            series_before, series_after = [], []
            for serie in series:
                serie_before, serie_after = serie.split_before(self.fixed_partitioning_inputs.split_date)
                series_before.append(serie_before)
                series_after.append(serie_after)

            return series_before, series_after

    def save_metrics(self, result_metrics, result_metrics6, output_folder):
        print(f'Average Metrics on the 6 locations \n'
              f'{dict(zip(self.metrics, np.mean(result_metrics6, axis=0)))}'
              )
        print(f'Average Metrics on the validation set ({len(result_metrics)} time series) \n'
              f'{dict(zip(self.metrics, np.mean(result_metrics, axis=0)))}'
              )
        if output_folder is not None:
            index = self.aquiferes_6 + \
                    [str(i) for i in range(len(result_metrics))]

            metrics_df = pd.DataFrame(data=np.concatenate((result_metrics6, result_metrics), axis=0),
                                      index=index, columns=self.metrics)
            metrics_df.to_csv(os.path.join(output_folder, f'global_metrics.csv'))

    def save_plots(self, serie_before, serie_after, predictions, aquifere_name, results, output_folder):
        serie_before.plot(label='target')
        serie_after.plot(label='truth')
        predictions.plot(label='predictions')

        plt.title(
            f'prediction_plot_{aquifere_name}'
            f'\n{dict(zip(self.metrics, np.round(results, 3)))}')
        plt.savefig(os.path.join(output_folder, 'plots/', f'prediction_plot_{aquifere_name}.png'),
                    bbox_inches="tight")
        plt.clf()
