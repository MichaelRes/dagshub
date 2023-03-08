import ast
import numpy as np
import random
import string
import itertools
from darts.utils.missing_values import fill_missing_values
from darts.dataprocessing.transformers import Scaler

def fill_missing_values_of_series(series):
    for i in range(len(series)):
        series[i] = fill_missing_values(series[i], fill='auto')
        series[i] = fill_missing_values(series[i], fill=0.0)

    return series


def create_combination(features):
    combinations = []
    for r in range(1, len(features) + 1):
        combinations.extend([list(x) for x in itertools.combinations(iterable=features, r=r)])
    return combinations


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def get_random_digits():
    return f'{np.random.randint(low=1, high=1e4):05d}'


def normalize_series(series):
    return [transform_serie(series[i]) for i in range(len(series))]


def transform_serie(serie):
    transformer = Scaler()
    return transformer.fit_transform(serie)


def transform_serie_and_return_scaler(serie):
    transformer = Scaler()
    return transformer.fit_transform(serie), transformer
