from src.dataloaders.csv_dataloader import PandasDataloader
from src.dataloaders.dill_dataloader import LocalTimeSeriesDillDataloader, GenericDillDataloader
import matplotlib.pyplot as plt
import os
import pandas as pd


class LocalPredictor:
    required_keys = ['model', 'best_features', 'fitted', 'location']

    def __init__(self,
                 model_dict: str,
                 dataloader: LocalTimeSeriesDillDataloader
                 ):
        self.model_dict = model_dict
        self.dataloader = dataloader
        self.data = dataloader.load()

    def predict(self, n_predict_points):
        predictions = self.model_dict['model'].predict(n=n_predict_points,
                                                       series=self.data['targets'][self.model_dict['location']],
                                                       past_covariates=self.data['covs'][self.model_dict['location']]
                                                       [self.model_dict['best_features']])
        return predictions

    def plot_and_save_predictions(self, predictions, output_folder):
        self.plot_predictions(predictions, output_folder)
        self.save_predictions(predictions, output_folder)

    def load_model(self):
        model_dict = GenericDillDataloader(os.path.split(self.model_path)[0],
                                           os.path.split(self.model_path)[1]).load()
        self.validate_input(model_dict)
        return model_dict

    @staticmethod
    def validate_input(model_dict):
        assert model_dict['fitted']
        assert all(key in model_dict.keys() for key in LocalPredictor.required_keys)

    def plot_predictions(self, predictions, output_folder):
        self.data['targets'][self.model_dict['location']].plot(label='pz')
        predictions.plot(label='predictions')
        plt.savefig(os.path.join(output_folder, self.model_dict['location'], f'predictions_plot.png'))
        plt.title(f"Predictions on {self.model_dict['location']}")
        plt.clf()

    def save_predictions(self, predictions, output_folder):
        df_results = pd.DataFrame(data=predictions.values(),
                                  index=predictions.time_index,
                                  columns=['predictions'])
        PandasDataloader.save(df_results, os.path.join(output_folder, self.model_dict['location'],
                                                    f'predictions.csv'))

