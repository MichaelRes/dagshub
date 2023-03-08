import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values


class LocalCleaner:
    year_shift = 365
    month_shift = 30
    week_shift = 7

    def __init__(self, min_date, max_date):
        self.min_date = min_date
        self.max_date = max_date

    def clean(self, data: pd.DataFrame):
        data_each_location_dict = self.compact_data_each_location(data)
        for location in data_each_location_dict.keys():
            data_each_location_dict[location] = self.format_date(data_each_location_dict[location])
            data_each_location_dict[location] = self.add_features(data_each_location_dict[location])
            data_each_location_dict[location] = self.between_selected_dates(data_each_location_dict[location])
        targets, covs = self.build_target_and_covs_all_location(data_each_location_dict)
        return {
            'targets': targets,
            'covs': covs
        }

    def generate_file_name(self):
        return f'data.pkl'

    @staticmethod
    def compact_data_each_location(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        dates = data['Date'].unique()
        locations = data['model_name'].unique().tolist()
        features = data['variable'].unique().tolist()
        cols = ['Date'] + features

        compact_df_each_location = dict.fromkeys(locations)

        for i, location in enumerate(locations):
            print(f'Processing location {location}')
            compact_data = np.zeros((len(dates), 1 + len(features)), dtype=object)
            compact_data[:, 0] = dates
            data_location = data[data['model_name'] == location]
            for j in range(len(dates)):
                data_row = data_location[data_location['Date'] == dates[j]]
                for k in range(len(features)):
                    try:
                        compact_data[j, 1 + k] = data_row.loc[data_row['variable'] == features[k], 'values'].values[0]
                    except:
                        compact_data[j, 1 + k] = np.nan

            compact_df_each_location[location] = pd.DataFrame(compact_data, columns=cols)

        return compact_df_each_location

    @staticmethod
    def format_date(data: pd.DataFrame) -> pd.DataFrame:
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
        return data

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.add_window_features(data)
        data = self.add_shift_features(data)
        data = self.add_ratio_features(data)
        data = self.add_pluie_dec_mars(data)
        return data

    def between_selected_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[data['Date'].between(self.min_date, self.max_date)]

    @staticmethod
    def build_target_and_covs_all_location(data_each_location_dict: Dict[str, pd.DataFrame],
                                           label='pz', date_col='Date') \
            -> Tuple[Dict[str, TimeSeries], Dict[str, TimeSeries]]:
        targets = dict()
        covs = dict()
        locations = sorted(data_each_location_dict.keys())

        for location in locations:
            target, cov = LocalCleaner.build_target_and_covs_one_location(data_each_location_dict[location], label, date_col)
            targets[location] = target
            covs[location] = cov
        return targets, covs

    @staticmethod
    def build_target_and_covs_one_location(data_location, label, date_col) -> Dict[str, TimeSeries]:
        features = data_location.columns.tolist()
        features.remove(date_col)
        features.remove(label)
        target = TimeSeries.from_dataframe(data_location, date_col, [label], fill_missing_dates=True, freq='1d')
        cov = TimeSeries.from_dataframe(data_location, date_col, features, fill_missing_dates=True, freq='1d')
        return LocalCleaner.fill_missing_values_of_series([target, cov])

    @staticmethod
    def fill_missing_values_of_series(timeseries: List[TimeSeries]):
        for idx in range(len(timeseries)):
            timeseries[idx] = fill_missing_values(timeseries[idx], fill='auto')
            timeseries[idx] = fill_missing_values(timeseries[idx], fill=0.0)
        return timeseries

    def add_ratio_features(self, data: pd.DataFrame):
        data['pluie_annuelle_ratio'] = data['pluie'].rolling(
            window=abs(self.year_shift)).mean() / data['pluie'].rolling(window=abs(10 * self.year_shift)).mean()
        data['etp_annuelle_ratio'] = data['etp'].rolling(
            window=abs(self.year_shift)).mean() / data['etp'].rolling(window=abs(10 * self.year_shift)).mean()
        return data

    def add_pluie_dec_mars(self, data: pd.DataFrame) -> pd.DataFrame:
        pluie_dec_mars_all_years = self.create_pluie_dec_mars_dict(data)

        def aux_func(row):
            if row.Date.month in [1, 2, 3] and row.Date.year - 1 in pluie_dec_mars_all_years.keys():
                return pluie_dec_mars_all_years[row.Date.year - 1]
            elif row.Date.month not in [1, 2, 3]:
                return pluie_dec_mars_all_years[row.Date.year]
            else:
                return 0

        data['pluie_dec_mars'] = data.apply(aux_func, axis=1)
        return data

    @staticmethod
    def create_pluie_dec_mars_dict(data: pd.DataFrame):
        res = {}
        for year in data['Date'].dt.year.unique():
            res[year] = data[(data['Date'].dt.year == year) & (data['Date'].dt.month.isin([12, 1, 2, 3]))]['pz'].mean()
        return res

    def add_window_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.add_window_features_weekly(data)
        data = self.add_window_features_monthly(data)
        data = self.add_window_features_yearly(data)
        return data

    def add_shift_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.add_shift_features_1y_ago(data)
        data = self.add_shift_features_2y_ago(data)
        return data

    def add_window_features_weekly(self, data: pd.DataFrame) -> pd.DataFrame:
        data['pluie_semaine'] = data['pluie'].rolling(window=self.week_shift).mean()
        data['etp_semaine'] = data['etp'].rolling(window=self.week_shift).mean()
        return data

    def add_window_features_monthly(self, data: pd.DataFrame) -> pd.DataFrame:
        data['pluie_mensuelle'] = data['pluie'].rolling(window=self.month_shift).mean()
        data['etp_mensuelle'] = data['etp'].rolling(window=self.month_shift).mean()
        return data

    def add_window_features_yearly(self, data: pd.DataFrame) -> pd.DataFrame:
        data['pluie_annuelle'] = data['pluie'].rolling(window=self.year_shift).mean()
        data['etp_annuelle'] = data['etp'].rolling(window=self.year_shift).mean()
        return data

    def add_shift_features_1y_ago(self, data: pd.DataFrame) -> pd.DataFrame:
        data['etp_lag_Y-1'] = data['etp'].shift(self.year_shift)
        data['pluie_lag_Y-1'] = data['pluie'].shift(self.year_shift)
        return data

    def add_shift_features_2y_ago(self, data: pd.DataFrame) -> pd.DataFrame:
        data['etp_lag_Y-2'] = data['etp'].shift(2*self.year_shift)
        data['pluie_lag_Y-2'] = data['pluie'].shift(2*self.year_shift)
        return data
