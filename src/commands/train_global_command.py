import logging
import os
from dataclasses import dataclass

from src.trainers.global_trainer import GlobalTrainer
from src.utils.utils import get_random_digits


@dataclass
class TrainGlobalCommand:
    trainer: GlobalTrainer
    result_folder: str
    logger = logging.getLogger(__name__)

    def __post_init__(self):
        os.makedirs(self.result_folder, exist_ok=True)
        self.output_folder = os.path.join(self.result_folder,
                                          f'global_train_{get_random_digits()}')
        os.makedirs(self.output_folder, exist_ok=True)

    def run(self):
        self.trainer.train(self.output_folder)
