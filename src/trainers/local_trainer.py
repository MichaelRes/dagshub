import darts
from darts.models.forecasting.forecasting_model import ForecastingModel
import os
from typing import List, Tuple
from src.dataloaders.dill_dataloader import LocalTimeSeriesDillDataloader, GenericDillDataloader

"""
This class allows to train one or many local models on all the training data.
"""


class LocalTrainer:

    def __init__(self,
                 models_paths: List[str],
                 dataloader: LocalTimeSeriesDillDataloader,
                 ):
        self.models_paths = models_paths
        self.models_dicts = self.load_models_dicts()
        self.dataloader = dataloader
        self.data = dataloader.load()
        self.locations = self.get_locations()

    def load_models_dicts(self):
        models_dicts = []
        for model_path in self.models_paths:
            models_dicts.append(GenericDillDataloader(os.path.split(model_path)[0],
                                                      os.path.split(model_path)[1]).load())

        return models_dicts

    def train(self, output_folder: str):
        if len(self.models_dicts) == 1 and self.models_dicts[0]['location'] == 'all_locations':
            model_dicts = self.train_semi_local(output_folder)
            return model_dicts
        else:
            self.check_models_dicts()
            self.train_local(output_folder)
            return self.models_dicts

    def train_local(self, output_folder: str):
        for idx, model_dict in enumerate(self.models_dicts):
            model_dict['model'].fit(series=self.data['targets'][model_dict['location']],
                                    past_covariates=self.data['covs'][model_dict['location']]
                                    [model_dict['best_features']])
            model_dict['fitted'] = True
            self.save_model(model_dict, os.path.join(output_folder, model_dict['location'],
                                                     f'model_fitted.pkl'))

    def train_semi_local(self, output_folder: str):
        semi_local_model = self.models_dicts[0]['model']
        features = self.models_dicts[0]['best_features']
        model_dicts = []
        for location in self.data['targets'].keys():
            local_model = semi_local_model.untrained_model()
            local_model.fit(series=self.data['targets'][location],
                            past_covariates=self.data['covs'][location][features])
            model_dict = {
                'model': local_model,
                'best_parameters': local_model.model_params,
                'best_features': features,
                'location': location,
                'fitted': True
            }

            self.save_model(model_dict, os.path.join(output_folder, location,
                                                     f'model_fitted.pkl'))
            model_dicts.append(model_dict)
        return model_dicts

    def check_models_dicts(self):
        for model_dict in self.models_dicts:
            assert model_dict['location'] != 'all_locations'

    @staticmethod
    def save_model(model_dict, output_file):
        GenericDillDataloader.save(model_dict,
                                   output_file)

    def get_locations(self):
        if len(self.models_dicts) == 1 and self.models_dicts[0]['location'] == 'all_locations':
            return self.data['targets'].keys()
        else:
            return [model_dict['location'] for model_dict in self.models_dicts]
