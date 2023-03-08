import logging
from dataclasses import dataclass

import pandas as pd


@dataclass
class PandasDataloader:
    filepath: str
    sep: str = ","
    logger = logging.getLogger(__name__)

    def load(self) -> pd.DataFrame:
        data = pd.read_csv(self.filepath, sep=self.sep)
        self.logger.info(f"Loaded dataset : {self.filepath}")
        return data

    @classmethod
    def save(cls, data_frame: pd.DataFrame, output_path: str):
        data_frame.to_csv(output_path)
        cls.logger.info(f"Saved dataset to {output_path}")
