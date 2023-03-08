import logging
import os
from dataclasses import dataclass
from typing import List

from src.dataloaders.dill_dataloader import LocalTimeSeriesDillDataloader, GenericDillDataloader
from src.predictors.local_predictor import LocalPredictor
from src.utils.utils import get_random_digits


@dataclass
class PredictCommand:
    model_paths: List[str]
    dataloader: LocalTimeSeriesDillDataloader
    result_folder: str
    n_predict_points: int
    logger = logging.getLogger(__name__)

    def __post_init__(self):
        self.model_dicts = self.load_model_dicts()
        self.locations = self.get_locations()
        os.makedirs(self.result_folder, exist_ok=True)

        self.output_folder = os.path.join(self.result_folder,
                                          f'preds_{get_random_digits()}')

        os.makedirs(self.output_folder, exist_ok=True)
        for location in self.locations:
            os.makedirs(os.path.join(self.output_folder, location), exist_ok=True)

    def run(self):
        self.generate_and_save_multiple_predictions(self.model_dicts)

    def generate_and_save_predictions(self, model_dict):
        predictor = LocalPredictor(model_dict, self.dataloader)
        predictions = predictor.predict(self.n_predict_points)
        predictor.plot_and_save_predictions(predictions, self.output_folder)

    def generate_and_save_multiple_predictions(self, model_dicts):
        for model_dict in model_dicts:
            self.generate_and_save_predictions(model_dict)

    def load_model_dicts(self):
        model_dicts = []
        for path in self.model_paths:
            model_dicts.append(GenericDillDataloader(os.path.split(path)[0],
                                                     os.path.split(path)[1]).load())
        return model_dicts

    def get_locations(self):
        return [model_dict['location'] for model_dict in self.model_dicts]
