from darts.models.forecasting.forecasting_model import ForecastingModel

import darts
from darts import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.utils.metrics import multiple_metrics

"""
This class allows to evaluate an unfitted forecasting model on all locations, using backtesting.
ONLY USED IN NOTEBOOKS
"""


class Evaluator:
    model: ForecastingModel  # model isn't trained here, just initialized
    data: dict
    start_date: pd.Timestamp
    past_covariates: dict
    n_metrics = 5

    def __init__(self, model, data, start_date=pd.Timestamp('2015-06-01'), past_covariates=None, forecast_horizon=182,
                 stride=365):
        self.model = model
        self.data = data
        self.start_date = start_date
        if past_covariates is None:
            self.past_covariates = dict(zip(self.data.keys(), [None for key in self.data.keys()]))
        else:
            self.past_covariates = past_covariates
        self.forecast_horizon = forecast_horizon
        self.stride = stride

    def evaluate_and_plot_rolling_window_all_locations(self, plot=False):
        global_backtest_metrics = np.zeros((self.n_metrics, len(self.data.keys())))
        for idx, location in enumerate(self.data.keys()):
            print(f'Metrics on location: {location}')
            global_backtest_metrics[:, idx] = self.evaluate_and_plot_rolling_window_one_location(location, plot)
            print(dict(zip(["MAPE", "sMAPE", "MSE", "MAE", "RMSE"], global_backtest_metrics[:, idx])))

        print(dict(zip(["MAPE", "sMAPE", "MSE", "MAE", "RMSE"], np.mean(global_backtest_metrics, axis=1))))
        print(dict(
            zip(["min MAPE", "min sMAPE", "min MSE", "min MAE", "min RMSE"], np.min(global_backtest_metrics, axis=1))))
        print(dict(
            zip(["max MAPE", "max sMAPE", "max MSE", "max MAE", "max RMSE"], np.max(global_backtest_metrics, axis=1))))
        return np.mean(global_backtest_metrics, axis=1)

    def evaluate_and_plot_rolling_window_one_location(self, location: str, plot=False):
        assert location in self.data.keys()
        try:
            backtest_metrics = \
                self.evaluate_and_plot_rolling_window_one_location_with_covs(self.past_covariates[location],
                                                                             location=location,
                                                                             plot=plot)
        except np.linalg.LinAlgError:
            print('Convergence issues, retrying with modified covariates data')
            covariates_array = self.past_covariates[location].values()
            covariates_array[covariates_array == 0] = 1e-20
            past_covariates_wo_zeros = TimeSeries.from_times_and_values(self.past_covariates[location].time_index,
                                                                        covariates_array)
            backtest_metrics = self.evaluate_and_plot_rolling_window_one_location_with_covs(past_covariates_wo_zeros,
                                                                                            location=location,
                                                                                            plot=plot)

        [backtest_mape, backtest_smape, backtest_mse, backtest_mae, backtest_rmse] = np.mean(backtest_metrics, axis=0)
        return [backtest_mape, backtest_smape, backtest_mse, backtest_mae, backtest_rmse]

    def evaluate_and_plot_rolling_window_one_location_with_covs(self, covs, location: str, plot=False):
        if plot:
            # historical forecast
            hist_forecasts = self.model.historical_forecasts(series=self.data[location], start=self.start_date,
                                                             forecast_horizon=self.forecast_horizon, stride=self.stride,
                                                             last_points_only=False,
                                                             past_covariates=covs)
            for hist in hist_forecasts:
                hist.plot(label=f'pz_{hist.time_index[0].year}')
            self.data[location].plot(label='truth')
            plt.show()

        # backtest_errors (not optimized, rewrite the backtest method if needed)
        backtest_metrics = self.model.backtest(series=self.data[location], start=self.start_date,
                                               forecast_horizon=self.forecast_horizon, stride=self.stride,
                                               last_points_only=False, metric=multiple_metrics,
                                               reduction=None, past_covariates=covs)
        return backtest_metrics
