import os

from src.dataloaders.dill_dataloader import GlobalTimeSeriesDillDataloader, GenericDillDataloader
import darts
from darts.models.forecasting.forecasting_model import ForecastingModel

from src.utils.model_inputs import ModelInputs


class GlobalTrainer:

    def __init__(self,
                 model_input: ModelInputs,
                 dataloader: GlobalTimeSeriesDillDataloader
                 ):
        self.model = model_input.model
        self.features = model_input.features
        data = dataloader.load()
        self.series = data['series']
        self.covs = [cov[self.features] for cov in data['covs']] if self.features != [] else None

    def train(self, output_folder):

        self.model.fit(series=self.series,
                       past_covariates=self.covs)
        self.save_model(os.path.join(output_folder, 'model.pkl'))

    def save_model(self, output_file):
        GenericDillDataloader.save(self.generate_model_json(),
                                   output_file)

    def generate_model_json(self):
        return {
            'model': self.model,
            'best_features': self.features,
            'location': 'global',
            'fitted': True
        }
