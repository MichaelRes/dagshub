"""
This cleaner allows to format a dataset.csv with columns `Date`, `model_name`,
`variable`, `values` into a pkl object that can be set as input to the global
prediction command
"""
import os

import numpy as np
import pandas as pd

from darts import TimeSeries
from src.cleaners.global_cleaner import GlobalCleaner
from src.cleaners.local_cleaner import LocalCleaner
from src.dataloaders.dill_dataloader import GenericDillDataloader
from src.utils.utils import fill_missing_values_of_series


class GlobalInputCleaner:

    def __init__(self, cov_path, long_and_lat_path, min_date, max_date, coordinates, date_col='Date', label='pz'):
        """
        :param coordinates: dict{str: Tuple[float, float]}
        dict containing the longitudes and latitudes of each locations
        """

        self.covs_df = pd.read_pickle(cov_path)
        long_and_lat_df = GenericDillDataloader(*os.path.split(long_and_lat_path)).load()

        self.long_array = long_and_lat_df['longitudes']
        self.lat_array = long_and_lat_df['latitudes']

        self.min_date = min_date
        self.max_date = max_date
        self.coordinates = coordinates

        self.date_col = date_col
        self.label = label

    def clean(self, data: pd.DataFrame):
        data_each_location_dict = self.compact_data_each_location(data, self.date_col, self.label)
        covs = []
        targets = []
        locations = sorted(list(data_each_location_dict.keys()))
        self.check_coordinates(locations)

        for location in locations:
            data_each_location_dict[location] = LocalCleaner.format_date(data_each_location_dict[location])
            data_each_location_dict[location] = self.between_selected_dates(data_each_location_dict[location])

            target = TimeSeries.from_dataframe(data_each_location_dict[location], self.date_col, [self.label],
                                               fill_missing_dates=True, freq='1d')
            [target] = fill_missing_values_of_series([target])
            cov = GlobalCleaner.build_cov_from_long_and_lat(self.covs_df,
                                                            self.coordinates[location][0],
                                                            self.coordinates[location][1],
                                                            self.long_array, self.lat_array)
            targets.append(target)
            covs.append(cov)
        return {'series': targets,
                'covs': covs}, locations

    @staticmethod
    def compact_data_each_location(data, date_col, target):
        locations = data['model_name'].unique().tolist()

        compact_df_each_location = dict.fromkeys(locations)

        for i, location in enumerate(locations):
            print(f'Processing location {location}')
            data_location = data[(data['model_name'] == location) & (data['variable'] == target)][[date_col, 'values']]
            compact_df_each_location[location] = pd.DataFrame(data_location.values, columns=['Date', target])

        return compact_df_each_location

    def between_selected_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[data['Date'].between(self.min_date, self.max_date)]

    def check_coordinates(self, locations):
        assert sorted(list((self.coordinates.keys()))) == locations
