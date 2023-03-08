import logging
import os
from dataclasses import dataclass

from src.cleaners.global_input_cleaner import GlobalInputCleaner
from src.dataloaders.csv_dataloader import PandasDataloader
from src.dataloaders.dill_dataloader import GenericDillDataloader
from src.predictors.global_predictor import GlobalPredictor
from src.utils.utils import get_random_digits


@dataclass
class PredictGlobalCommand:
    model_path: str
    dataset_path: str
    input_cleaner: GlobalInputCleaner
    result_folder: str
    n_predict_points: int
    logger = logging.getLogger(__name__)

    def __post_init__(self):
        self.model_dict = self.load_model_dict()

        os.makedirs(self.result_folder, exist_ok=True)
        self.output_folder = os.path.join(self.result_folder,
                                          f'global_preds_{get_random_digits()}')
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, 'plots'), exist_ok=True)

    def run(self):
        data = PandasDataloader(self.dataset_path, sep=";").load()
        data_dict, locations = self.input_cleaner.clean(data)
        predictor = GlobalPredictor(self.model_dict, data_dict)
        predictions = predictor.predict(self.n_predict_points)
        predictor.save_and_plot_predictions(predictions, self.output_folder, locations)

    def load_model_dict(self):
        return GenericDillDataloader(os.path.split(self.model_path)[0],
                                     os.path.split(self.model_path)[1]).load()
