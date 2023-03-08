import logging
import os
from dataclasses import dataclass

import warnings

from src.predictors.local_predictor import LocalPredictor
from src.trainers.local_trainer import LocalTrainer
from src.utils.utils import get_random_digits

warnings.filterwarnings(action='ignore', category=FutureWarning)


@dataclass
class TrainCommand:
    trainer: LocalTrainer
    result_folder: str
    n_predict_points: int
    logger = logging.getLogger(__name__)

    def __post_init__(self):
        os.makedirs(self.result_folder, exist_ok=True)

        self.output_folder = os.path.join(self.result_folder,
                                          f'training_{get_random_digits()}')

        os.makedirs(self.output_folder, exist_ok=True)
        for location in self.trainer.locations:
            os.makedirs(os.path.join(self.output_folder, location), exist_ok=True)

    def run(self):
        model_dicts = self.trainer.train(self.output_folder)
        self.generate_and_save_multiple_predictions(model_dicts)

    def generate_and_save_predictions(self, model_dict):
        predictor = LocalPredictor(model_dict, self.trainer.dataloader)
        predictions = predictor.predict(self.n_predict_points)
        predictor.plot_and_save_predictions(predictions, self.output_folder)

    def generate_and_save_multiple_predictions(self, model_dicts):
        for model_dict in model_dicts:
            self.generate_and_save_predictions(model_dict)
