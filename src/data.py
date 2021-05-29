from typing import Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler

# years
train_list = [
    1992,
    2015,
    2017,
    1996,
    2010,
    2019,
    1987,
    1990,
    1991,
    2000,
    2006,
    1999,
    2009,
    2014,
    2008,
    1988,
    1986,
    1985,
]
val_list = [1994, 2007, 2018, 2011, 2016, 1998, 1995, 2002]
test_list = [2001, 2003, 2005, 2012, 2013, 1989, 1993, 1997, 2004]


class LenaDataset(Dataset):
    def __init__(self, label_df, full_df, cat_cols, num_cols, last_day_previous_year=365 - (31 + 30 + 31)):
        self.label_df = label_df.copy()
        self.full_df = full_df.copy()

        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)

        self.last_day_previous_year = last_day_previous_year

    def __len__(self):
        return self.label_df.shape[0]

    def __getitem__(self, idx):
        station = self.label_df.station_id.values[idx]
        year = self.label_df.year.values[idx]
        day = self.label_df.day.values[idx]
        target = self.label_df.ice_jam.values[idx]
        prev_year = year - 1
        part_index1 = (self.full_df.year == prev_year) & (self.full_df.day > self.last_day_previous_year)
        part_index2 = (self.full_df.year == year) & (self.full_df.day < 31 + 30 + 31)
        part_index3 = self.full_df.hydro_station_id == station
        any_id = self.full_df.loc[part_index3, "hydro_fixed_station_id_categorical"].values[0]
        feat_matrix = self.full_df.loc[(part_index1 | part_index2) & part_index3, self.cat_cols + self.num_cols].values
        new_feat_matrix = np.zeros((139, feat_matrix.shape[1]))
        new_feat_matrix[-feat_matrix.shape[0] :] = feat_matrix

        return {"x": new_feat_matrix, "y": target, "day": day, "station": any_id}


batch_size = 128


class LenaDataModule(pl.LightningDataModule):
    def __init__(self, train, test, features_df, batch_size=128, num_workers=2):
        super().__init__()

        self.train = train
        self.test = test
        self.features_df = features_df
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.numerical_features = [c for c in features_df if c.endswith("numeric")]
        self.categorical_features = [c for c in features_df if c.endswith("categorical")]

        print("Numerical Features")
        print(self.numerical_features)
        print("Categorical Features")
        print(self.categorical_features)

        self.train_ds = LenaDataset(
            self.train.loc[self.train.year.isin(train_list)],
            self.features_df,
            self.categorical_features,
            self.numerical_features,
        )
        self.valid_ds = LenaDataset(
            self.train.loc[self.train.year.isin(val_list)],
            self.features_df,
            self.categorical_features,
            self.numerical_features,
        )
        self.test_ds = LenaDataset(self.test, self.features_df, self.categorical_features, self.numerical_features)

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        train_sampler = RandomSampler(self.train_ds)
        return DataLoader(self.train_ds, sampler=train_sampler, batch_size=batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        valid_sampler = SequentialSampler(self.valid_ds)
        return DataLoader(self.valid_ds, sampler=valid_sampler, batch_size=batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        test_sampler = SequentialSampler(self.test_ds)
        return DataLoader(self.test_ds, sampler=test_sampler, batch_size=batch_size, num_workers=self.num_workers)
