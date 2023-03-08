import logging
import os
import pandas as pd
import numpy as np
import warnings
from typing import List
import matplotlib.pyplot as plt
from dataclasses import dataclass

from src.dataloaders.dill_dataloader import GenericDillDataloader
from src.evaluators.walk_forward_evaluator import WalkForwardEvaluator
from src.utils.utils import get_random_digits

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


@dataclass
class WalkForwardCommand:
    evaluator: WalkForwardEvaluator
    result_folder: str
    start_dates: List[str]
    save_model: bool
    logger = logging.getLogger(__name__)

    def __post_init__(self):
        os.makedirs(self.result_folder, exist_ok=True)

        self.output_folder = os.path.join(self.result_folder,
                                          f'eval_{get_random_digits()}')
        for location in self.evaluator.locations:
            os.makedirs(os.path.join(self.output_folder, location), exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

    def run(self):
        if len(self.start_dates) == 1:
            self.run_on_start_date(self.start_dates[0], self.output_folder)
            self.evaluator.save_metrics(self.output_folder)

        else:
            avg_metrics_by_period = np.zeros((len(self.start_dates), len(self.evaluator.metrics)))
            for idx, start_date in enumerate(self.start_dates):
                avg_metrics_by_period[idx] = self.run_on_start_date(start_date)
            self.plot_results(avg_metrics_by_period)
            self.save_results(avg_metrics_by_period)
        self.save_model_if_required()

    def run_on_start_date(self, start_date: str,
                          output_folder: str = None) -> np.array:
        self.evaluator.set_start_date(start_date)
        global_backtest_metrics_mean = self.evaluator.evaluate_and_plot_rolling_window_all_locations(output_folder)
        return global_backtest_metrics_mean

    def plot_results(self, avg_metrics_by_period):
        x = self.start_dates
        for metric_idx in range(len(self.evaluator.metrics)):
            y = avg_metrics_by_period[:, metric_idx]
            plt.plot(x, y, label=self.evaluator.metrics[metric_idx])
            plt.xlabel('Starting date')
            plt.ylabel(f'{self.evaluator.metrics[metric_idx]}')
            plt.title(f'Average {self.evaluator.metrics[metric_idx]} as a function of the forecasting periods')
            # TODO improve naming
            plt.savefig(os.path.join(self.output_folder,
                                     f'{self.evaluator.metrics[metric_idx]}_{len(self.start_dates)}_periods_.png'),
                        bbox_inches="tight")

    def save_results(self, avg_metrics_by_period):
        metrics_df = pd.DataFrame(data=avg_metrics_by_period, index=self.start_dates, columns=self.evaluator.metrics)
        metrics_df.to_csv(
            os.path.join(self.output_folder, f'metrics_6periods.csv'))

    def save_model_if_required(self):
        if self.save_model:
            outputs_json = self.evaluator.get_model_json()
            GenericDillDataloader.save(outputs_json,
                                       os.path.join(self.output_folder,
                                                    'model.pkl'))
