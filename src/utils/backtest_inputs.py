from dataclasses import dataclass
import pandas as pd
from src.utils.metrics import multiple_metrics, multiple_metrics_global


@dataclass
class BacktestInputs:

    def __init__(self, start_date, forecast_horizon, stride, metric=multiple_metrics):
        self.start_date = pd.Timestamp(start_date)
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.metric = metric

    def from_start_date(self, start_date):
        self.start_date = pd.Timestamp(start_date)
        return BacktestInputs(start_date, self.forecast_horizon, self.stride, self.metric)


@dataclass
class FixedPartitioningInputs:

    def __init__(self, split_date, forecast_horizon, metric=multiple_metrics_global):
        self.split_date = pd.Timestamp(split_date)
        self.forecast_horizon = forecast_horizon
        self.metric = metric
