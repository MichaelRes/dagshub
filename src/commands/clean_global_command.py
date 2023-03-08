import logging
import os
from dataclasses import dataclass

import pandas as pd

from src.cleaners.global_cleaner import GlobalCleaner
from src.dataloaders.csv_dataloader import PandasDataloader
from src.dataloaders.dill_dataloader import GlobalTimeSeriesDillDataloader, LocalTimeSeriesDillDataloader, \
    GenericDillDataloader


@dataclass
class CleanGlobalCommand:
    cleaner: GlobalCleaner
    dataset_path: str
    aquiferes_folder_path: str
    targets_and_covs_6_locations_path: str
    cov_path: str
    long_and_lat_path: str
    output_folder: str
    filter_bdlisa: str = None
    logger = logging.getLogger(__name__)

    def __post_init__(self):
        os.makedirs(self.output_folder, exist_ok=True)

    def run(self):
        covs_df = pd.read_pickle(self.cov_path)
        long_and_lat_df = GenericDillDataloader(*os.path.split(self.long_and_lat_path)).load()
        long_array = long_and_lat_df['longitudes']
        lat_array = long_and_lat_df['latitudes']
        self.run_global(covs_df, long_array, lat_array, filter_bdlisa=self.filter_bdlisa)
        if self.targets_and_covs_6_locations_path is not None:
            self.run_6_locations(covs_df, long_array, lat_array)

    def run_global(self, covs_df, long_array, lat_array, filter_bdlisa):
        output_filename = self.cleaner.generate_file_name(filter_bdlisa)
        data = PandasDataloader(self.dataset_path, sep=";").load()
        target_and_covs_dict = self.cleaner.clean(data, self.aquiferes_folder_path, covs_df,
                                                  long_array, lat_array, filter_bdlisa)
        GlobalTimeSeriesDillDataloader.save(target_and_covs_dict, os.path.join(self.output_folder, output_filename))

    def run_6_locations(self, covs_df, long_array, lat_array):
        output_filename_6_locations = self.cleaner.generate_file_name_6_locations()
        data_6_locations = LocalTimeSeriesDillDataloader(*os.path.split(self.targets_and_covs_6_locations_path)).load()
        target_and_covs_6_locations_dict = self.cleaner.clean_6_locations(data_6_locations, covs_df, long_array,
                                                                          lat_array)
        GlobalTimeSeriesDillDataloader.save(target_and_covs_6_locations_dict,
                                            os.path.join(self.output_folder, output_filename_6_locations))


@dataclass
class CleanGlobalCovariatesCommand:
    cov_folder: str
    output_folder: str

    def __post_init__(self):
        os.makedirs(self.output_folder, exist_ok=True)

    def run(self):
        covs_df = GlobalCleaner.clean_global_covariates(self.cov_folder)
        long_and_lat_dict = GlobalCleaner.get_long_and_lat_array(self.cov_folder)
        covs_df.to_pickle(os.path.join(self.output_folder, 'global_covariates.pkl'))
        GenericDillDataloader.save(long_and_lat_dict, os.path.join(self.output_folder, 'long_and_lat.pkl'))
