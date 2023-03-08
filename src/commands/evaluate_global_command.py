import logging
import os
from dataclasses import dataclass

from src.evaluators.global_evaluator import GlobalEvaluator
from src.utils.utils import get_random_digits


@dataclass
class EvaluateGlobalCommand:
    evaluator: GlobalEvaluator
    result_folder: str
    logger = logging.getLogger(__name__)

    def __post_init__(self):
        os.makedirs(self.result_folder, exist_ok=True)
        self.output_folder = os.path.join(self.result_folder,
                                          f'global_eval_{get_random_digits()}')

        os.makedirs(os.path.join(self.output_folder, 'plots/'), exist_ok=True)

    def run(self):
        self.evaluator.evaluate(self.output_folder)
