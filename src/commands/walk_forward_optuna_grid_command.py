import logging
import os
from dataclasses import dataclass
from src.trainers.walk_forward_optuna_grid_trainer import WalkForwardOptunaGridSearchTrainer
from src.evaluators.walk_forward_evaluator import WalkForwardEvaluator
from src.dataloaders.dill_dataloader import GenericDillDataloader
import warnings

from src.utils.model_inputs import ModelInputs
from src.utils.utils import get_random_string, get_random_digits

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


@dataclass
class WalkForwardOptunaGridCommand:
    trainer: WalkForwardOptunaGridSearchTrainer
    result_folder: str
    each: bool = True
    logger = logging.getLogger(__name__)

    def __post_init__(self):
        os.makedirs(self.result_folder, exist_ok=True)
        self.output_folder = os.path.join(self.result_folder,
                                          'grid_' + get_random_digits())
        os.makedirs(self.output_folder, exist_ok=True)
        for location in self.trainer.locations:
            os.makedirs(os.path.join(self.output_folder, location), exist_ok=True)
        if not self.each:
            os.makedirs(os.path.join(self.output_folder, 'all_locations'), exist_ok=True)

    def run(self):
        if not self.each:
            self.run_on_all_locations()
        else:
            self.run_on_each_location()

    def run_on_all_locations(self):
        best_model, best_parameters, best_features = self.trainer.find_best_hyperparameters_all_locations()
        self.evaluate_on_locations_and_save(best_model, best_features, self.trainer.locations)
        #TODO add plot when len(start_dates) > 1 like in eval command

    def run_on_each_location(self):
        best_models_each_location, _, best_features_each_location = \
            self.trainer.find_best_hyperparameters_each_location()
        self.evaluate_and_save_for_each_locations(best_models_each_location,
                                                  best_features_each_location)
        #TODO add plot when len(start_dates) > 1 like in eval command

    def evaluate_on_locations_and_save(self, best_model, best_features, locations):
        evaluator = self.get_evaluator_for_model(ModelInputs(best_model, best_features), locations)
        evaluator.evaluate_and_plot_rolling_window_all_locations(self.output_folder)
        self.save_outputs(evaluator.get_model_json())

    def evaluate_and_save_for_each_locations(self, best_models_each_location,
                                             best_features_each_location):
        for i in range(len(best_models_each_location)):
            self.evaluate_on_locations_and_save(best_models_each_location[i],
                                                best_features_each_location[i],
                                                [self.trainer.locations[i]])

    def get_evaluator_for_model(self, model_input, locations):
        return WalkForwardEvaluator(model_input, self.trainer.dataloader, self.trainer.backtest_inputs,
                                    locations=locations)

    def save_outputs(self, outputs_json):
        GenericDillDataloader.save(outputs_json,
                                   os.path.join(self.output_folder, outputs_json['location'],
                                                f'best_model.pkl'))
