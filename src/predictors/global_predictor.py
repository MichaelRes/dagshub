import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.dataloaders.csv_dataloader import PandasDataloader
from src.dataloaders.dill_dataloader import GlobalTimeSeriesDillDataloader, GenericDillDataloader


class GlobalPredictor:

    def __init__(self,
                 model_dict: dict,
                 data_dict: dict
                 ):
        self.model = model_dict['model']
        self.features = model_dict['best_features']
        self.series = data_dict['series']
        self.covs = [cov[self.features] for cov in data_dict['covs']] if self.features != [] else None

    def predict(self, n_points):
        preds = self.model.predict(n=n_points,
                                   series=self.series,
                                   past_covariates=self.covs)
        return preds

    def save_and_plot_predictions(self, predictions, output_folder, locations):
        self.plot_predictions(predictions, output_folder, locations)
        self.save_predictions(predictions, output_folder, locations)

    def plot_predictions(self, predictions, output_folder, locations):
        for i, prediction in enumerate(predictions):
            self.series[i].plot(label='pz')
            prediction.plot(label='predictions')
            plt.title(f"Predictions on location {locations[i]}")
            plt.savefig(os.path.join(output_folder, 'plots', f'predictions_{locations[i]}.png'),
                        box_inches="tight")
            plt.clf()

    @staticmethod
    def save_predictions(predictions, output_folder, locations):
        df_results = pd.DataFrame(data=np.array([p.values().squeeze() for p in predictions]).T,
                                  index=predictions[0].time_index,
                                  columns=[f'{locations[i]}' for i in range(len(predictions))])
        PandasDataloader.save(df_results, os.path.join(output_folder,
                                                       'predictions.csv'))

    def load_model(self):
        model = GenericDillDataloader(os.path.split(self.model_path)[0],
                                      os.path.split(self.model_path)[1]).load()

        return model
