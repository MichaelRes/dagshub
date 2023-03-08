import dill
import logging
import os
from typing import Dict, List, Union
from dataclasses import dataclass
from darts import TimeSeries


@dataclass
class LocalTimeSeriesDillDataloader:
    folder_path: str
    filename: str
    logger = logging.getLogger(__name__)

    def load(self) -> Dict[str, Dict[str, TimeSeries]]:
        filepath = os.path.join(self.folder_path, self.filename)
        with open(filepath, "rb") as outfile:
            data = dill.load(outfile)
        self.logger.info(f"Loaded the time series data from : {filepath}")
        return data

    @classmethod
    def save(cls, data: Dict[str, Dict[str, TimeSeries]], filepath: str):
        with open(os.path.join(filepath), "wb") as outfile:
            dill.dump(data, outfile)
        cls.logger.info(f"Saved the time series data to : {filepath}")


@dataclass
class GlobalTimeSeriesDillDataloader:
    folder_path: str
    filename: str
    logger = logging.getLogger(__name__)

    def load(self) -> Dict[str, Union[TimeSeries, List[TimeSeries]]]:
        filepath = os.path.join(self.folder_path, self.filename)
        with open(filepath, "rb") as outfile:
            data = dill.load(outfile)
        self.logger.info(f"Loaded the time series data from : {filepath}")
        return data

    @classmethod
    def save(cls, data: Dict[str, Union[TimeSeries, List[TimeSeries]]], filepath: str):
        with open(os.path.join(filepath), "wb") as outfile:
            dill.dump(data, outfile)
        cls.logger.info(f"Saved the time series data to : {filepath}")


@dataclass
class GenericDillDataloader:
    folder_path: str
    filename: str
    logger = logging.getLogger(__name__)

    def load(self) -> Dict:
        filepath = os.path.join(self.folder_path, self.filename)
        with open(filepath, "rb") as outfile:
            data = dill.load(outfile)
        self.logger.info(f"Loaded from : {filepath}")
        return data

    @classmethod
    def save(cls, data: Dict, filepath: str):
        with open(os.path.join(filepath), "wb") as outfile:
            dill.dump(data, outfile)
        cls.logger.info(f"Saved to : {filepath}")
