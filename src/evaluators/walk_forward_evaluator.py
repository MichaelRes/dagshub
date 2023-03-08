from darts import TimeSeries
import os
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from src.dataloaders.dill_dataloader import LocalTimeSeriesDillDataloader
from src.utils.backtest_inputs import BacktestInputs
from src.utils.model_inputs import ModelInputs

"""
This class allows to evaluate an unfitted forecasting model on all locations, using backtesting.
"""


class WalkForwardEvaluator:
    def __init__(self, model_input: ModelInputs,
                 dataloader: LocalTimeSeriesDillDataloader,
                 backtest_inputs: BacktestInputs,
                 locations=None,
                 ):
        self.model = model_input.model
        self.data = dataloader.load()
        self.features = model_input.features if model_input.features is not None \
            else list(list(self.data['covs'].values())[0].components)
        self.backtest_inputs = backtest_inputs
        self.locations = locations if locations is not None else list(self.data['targets'].keys())
        self.metrics = ["MAPE", "sMAPE", "MSE", "MAE", "RMSE"]
        self.result_metrics = None

    def set_start_date(self, start_date):
        self.backtest_inputs = self.backtest_inputs.from_start_date(start_date)

    def evaluate_and_plot_rolling_window_all_locations(self, output_folder: str = None) -> np.array:
        global_backtest_metrics = np.zeros((len(self.locations), len(self.metrics)))
        for idx, location in enumerate(self.locations):
            global_backtest_metrics[idx, :] = self.evaluate_and_plot_rolling_window_one_location(location,
                                                                                                 output_folder)
        self.set_result_metrics(global_backtest_metrics)
        return np.mean(global_backtest_metrics, axis=0)

    def evaluate_and_plot_rolling_window_one_location(self, location: str, output_folder: str) -> np.array:
        assert location in self.locations
        try:
            results = self.evaluate_with_targets_and_covs(self.data['targets'][location],
                                                          self.data['covs'][location][self.features],
                                                          location,
                                                          output_folder)
        except np.linalg.LinAlgError:
            print('Convergence issues, retrying with modified covariates data')
            past_covariates_wo_zeros = self.remove_zeros_from_covariates(self.data['covs'][location][self.features])
            results = self.evaluate_with_targets_and_covs(self.data['targets'][location],
                                                          past_covariates_wo_zeros,
                                                          location,
                                                          output_folder)
        return results

    def evaluate_with_targets_and_covs(self, target: TimeSeries,
                                       covs: TimeSeries,
                                       location: str,
                                       output_folder: str) -> np.array:
        backtest_metrics = self.model.backtest(series=target, start=self.backtest_inputs.start_date,
                                               forecast_horizon=self.backtest_inputs.forecast_horizon,
                                               stride=self.backtest_inputs.stride,
                                               last_points_only=False, metric=self.backtest_inputs.metric,
                                               reduction=None, past_covariates=covs)

        results = np.mean(backtest_metrics, axis=0)

        if output_folder is not None:
            self.plot_and_save(target, covs, location, results, output_folder)

        return results

    def set_result_metrics(self, global_backtest_metrics):
        self.result_metrics = global_backtest_metrics

    def save_metrics(self, output_folder):
        self.print_metrics()
        metrics_df = pd.DataFrame(data=self.result_metrics, index=self.locations, columns=self.metrics)
        metrics_df.to_csv(os.path.join(output_folder, f'metrics.csv'))
        return metrics_df

    def print_metrics(self):
        """
        :param global_backtest_metrics: array(n_metrics, n_locations)
        """
        print(dict(zip(["MAPE", "sMAPE", "MSE", "MAE", "RMSE"], np.mean(self.result_metrics, axis=0))))
        mlflow.log_metrics(dict(zip(["MAPE", "sMAPE", "MSE", "MAE", "RMSE"], np.mean(self.result_metrics, axis=0))))

    @staticmethod
    def remove_zeros_from_covariates(covs_location: TimeSeries):
        covariates_array = covs_location.values()
        covariates_array[covariates_array == 0] = 1e-20
        covariates_wo_zeros = TimeSeries.from_times_and_values(covs_location.time_index,
                                                               covariates_array)
        return covariates_wo_zeros

    def plot_and_save(self, target, covs, location, results, output_folder):
        hist_forecasts = self.model.historical_forecasts(series=target,
                                                         start=self.backtest_inputs.start_date,
                                                         forecast_horizon=self.backtest_inputs.forecast_horizon,
                                                         stride=self.backtest_inputs.stride,
                                                         last_points_only=False,
                                                         past_covariates=covs)
        fig = plt.figure()
        for hist in hist_forecasts:
            hist.plot(label=f'pz_{hist.time_index[0].year}')
        target.plot(label='truth')
        plt.title(
            f'Historical Forecast in {location}\n{dict(zip(["MAPE", "sMAPE", "MSE", "MAE", "RMSE"], np.round(results, 3)))}')
        # TODO improve naming
        plt.savefig(os.path.join(output_folder, location, 'historical_forecast_plot.png'),
                    bbox_inches="tight")
        mlflow.log_figure(fig, "plots/hist_forecast.png")
        plt.clf()

    def get_model_json(self):
        outputs_json = {
            'model': self.model,
            'best_parameters': self.model.model_params,
            'best_features': self.features,
            'location': self.locations[0] if len(self.locations) == 1 else 'all_locations',
            'fitted': False
        }
        return outputs_json

