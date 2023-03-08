import numpy as np
import pandas as pd
from darts import TimeSeries
import netCDF4
import os

from src.utils.utils import fill_missing_values_of_series


class GlobalCleaner:
    year_shift = 365
    month_shift = 30
    week_shift = 7

    # longitude / latitude
    aquiferes = {
        'bioule': [1.5376902, 44.0897873],
        'les_barthes': [1.170013, 44.096339],
        'saint_felix': [1.888, 43.449],
        'saint_porquier': [1.1772799491, 44.0043983459],
        'tarsac': [-0.1167, 43.6667],
        'verniolles': [1.649, 43.081]
    }

    def __init__(self, min_date, max_date):
        self.min_date = min_date
        self.max_date = max_date

    def clean(self, aquiferes_df, aquiferes_folder_path, covs_df, long_array, lat_array, filter_bdlisa):
        aquiferes_df_subset = self.select_subset_of_aquiferes(aquiferes_df, filter_bdlisa)
        series, covs = self.retrieve_time_series_from_aquiferes(aquiferes_df_subset, aquiferes_folder_path, covs_df,
                                                                long_array, lat_array)
        print(f'{len(series)} Timeseries have been retrieved')
        return {'series': series,
                'covs': covs
                }

    def clean_6_locations(self, targets_and_covs_per_location_dict, covs_df, long_array, lat_array):
        locations = sorted(targets_and_covs_per_location_dict['targets'].keys())
        covs = self.build_covs_6_locations(locations, covs_df, long_array, lat_array)
        return {'series': [targets_and_covs_per_location_dict['targets'][key] for key in locations],
                'covs': covs
                }

    def generate_file_name(self, filter_bdlisa):
        file_str = f'_bdlisa_{filter_bdlisa}' if filter_bdlisa is not None else ''
        return f'all_aquiferes_between_{self.min_date}_and_{self.max_date}{file_str}.pkl'

    def generate_file_name_6_locations(self):
        return f'aquiferes_6_locations_between_{self.min_date}_and_{self.max_date}.pkl'

    def select_subset_of_aquiferes(self, aquiferes_df, filter_bdlisa):
        aquiferes_df = self.subset_date(aquiferes_df)
        if filter_bdlisa is not None:
            aquiferes_df = self.subset_bdlisa(aquiferes_df, filter_bdlisa)
        return aquiferes_df

    def retrieve_time_series_from_aquiferes(self, aquiferes_df, aquiferes_folder_path, covs_df, long_array, lat_array):
        series = []
        covs = []
        files_with_errors = []
        for i in range(len(aquiferes_df)):
            try:
                filename = aquiferes_df.iloc[i].code_bss.replace('/', '#') + '#' + aquiferes_df.iloc[i].nom_commune
                df_pz = pd.read_csv(aquiferes_folder_path + filename + '.csv', delimiter=';')
                df_pz = self.clean_one_aquifere(df_pz)
                serie = self.build_serie_from_df(df_pz)
                cov = self.build_cov_from_long_and_lat(covs_df,
                                                       aquiferes_df.iloc[i].x,
                                                       aquiferes_df.iloc[i].y,
                                                       long_array,
                                                       lat_array)

                # check if we can split
                assert serie.time_index.min() < pd.Timestamp(self.min_date)
                assert serie.time_index.max() > pd.Timestamp(self.max_date)
                assert cov.time_index.min() < pd.Timestamp(self.min_date)
                assert cov.time_index.max() > pd.Timestamp(self.max_date)

                series.append(serie)
                covs.append(cov)

            except (AssertionError, FileNotFoundError):
                files_with_errors.append(aquiferes_df.iloc[i].code_bss.replace('/', '#')
                                         + '#' + aquiferes_df.iloc[i].nom_commune)
                pass

        print(f'{len(files_with_errors)} Files with errors: {files_with_errors[:10]}')

        return series, covs

    @staticmethod
    def clean_one_aquifere(df):
        df = df.drop_duplicates(subset=['date_mesure'])
        if df['niveau_nappe_eau'].min() < 0:
            df['niveau_nappe_eau'] = df['niveau_nappe_eau'] + abs(df['niveau_nappe_eau'].min()) + 1
        return df

    def subset_date(self, aquiferes_df):
        aquiferes_df['date_debut_mesure'] = pd.to_datetime(aquiferes_df['date_debut_mesure'], format='%Y-%m-%d')
        aquiferes_df['date_fin_mesure'] = pd.to_datetime(aquiferes_df['date_fin_mesure'], format='%Y-%m-%d')
        return aquiferes_df[(aquiferes_df['date_debut_mesure'] < self.min_date) & (
                aquiferes_df['date_fin_mesure'] > self.max_date)]

    def subset_bdlisa(self, aquiferes_df, filter_bdlisa):
        return aquiferes_df[aquiferes_df['codes_bdlisa'].str[0] == filter_bdlisa]

    def build_serie_from_df(self, df_pz):
        serie = TimeSeries.from_dataframe(df_pz, 'date_mesure', 'niveau_nappe_eau', fill_missing_dates=True,
                                          freq='1d')
        [serie] = fill_missing_values_of_series([serie])
        return serie

    @staticmethod
    def build_cov_from_long_and_lat(covs_df, long, lat, long_array, lat_array):
        long_index, lat_index = GlobalCleaner.find_index_from_long_and_lat(long, lat, long_array, lat_array)
        cov = covs_df.apply(lambda row: row.apply(lambda col: col[lat_index, long_index]))

        # TODO Add more features ? If so, add a features attribute in the evaluators and trainers
        cov = TimeSeries.from_dataframe(cov, value_cols=['pev', 'tp', 't2m'], fill_missing_dates=True,
                                        freq='1d')
        [cov] = fill_missing_values_of_series([cov])
        return cov

    @staticmethod
    def find_index_from_long_and_lat(long, lat, long_array, lat_array):
        long_index = np.absolute(long_array - long).argmin()
        lat_index = np.absolute(lat_array - lat).argmin()

        return long_index, lat_index

    @staticmethod
    def clean_global_covariates(path):
        covs = pd.DataFrame({
            'date': [],
            'pev': [],
            'tp': [],
            't2m': []
        }, dtype=object)
        files = os.listdir(path)
        for count, file in enumerate(files):
            try:
                data = netCDF4.Dataset(os.path.join(path, file))
                index_range = len(data.variables['time'][:]) // 24
                for i in range(index_range):
                    covs = covs.append({
                        'date': pd.Timestamp('1900-01-01') + pd.DateOffset(
                            hours=int(data.variables['time'][:][i * 24])),
                        'pev': data.variables['pev'][i * 24, :, :].data,
                        'tp': data.variables['tp'][i * 24, :, :].data,
                        't2m': data.variables['t2m'][i * 24, :, :].data,
                    }, ignore_index=True)
            except OSError:
                pass
        return covs.sort_values('date').set_index('date')

    @staticmethod
    def get_long_and_lat_array(path):
        files = os.listdir(path)
        count = 0
        long_array = None
        while long_array is None:
            try:
                cov = netCDF4.Dataset(os.path.join(path, files[count]))
                long_array = cov.variables['longitude'][:].data
                lat_array = cov.variables['latitude'][:].data

            except OSError:
                pass

        return {'longitudes': long_array,
                'latitudes': lat_array}

    def build_covs_6_locations(self, locations, covs_df, long_array, lat_array):
        covs = []
        for location in locations:
            long, lat = self.aquiferes[location]
            cov = self.build_cov_from_long_and_lat(covs_df, long, lat, long_array, lat_array)
            covs.append(cov)
        return covs
