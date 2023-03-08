from dataclasses import dataclass
from typing import List
from darts.models.forecasting.forecasting_model import ForecastingModel


@dataclass
class ModelInputs:

    def __init__(self, model: ForecastingModel, features: List[str]):
        self.model = model
        self.features = features
