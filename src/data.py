from datetime import date, timedelta
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
        self.full_df = full_df.drop(columns=[c for c in full_df.columns if "namask" in c or "_pred" in c]).copy()

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


class LenaDatasetExtra(Dataset):
    def __init__(
        self,
        full_df,
        cat_cols,
        num_cols,
        target_cols,
        gap=66,
        history=139,
        mode="train",
    ):
        self.mode = mode
        self.full_df = full_df.drop(columns=[c for c in full_df.columns if "namask" in c or "_pred" in c]).copy()

        if mode == "train":
            self.target_df = self.full_df[(self.full_df["day"] >= 31) & (self.full_df["day"] <= 150)]
        elif mode == "val":
            self.target_df = self.full_df[(self.full_df["day"] >= 111) & (self.full_df["day"] <= 111 + 45)]
        elif mode == "test":
            self.target_df = self.full_df[(self.full_df["day"] >= 111) & (self.full_df["day"] <= 111 + 45)]

        def make_full_date(x):
            year, day = x
            return date(year=year, month=1, day=1) + timedelta(days=day - 1)

        self.full_df["full_date"] = self.full_df[["year", "day"]].apply(make_full_date, axis=1)
        self.target_df["full_date"] = self.target_df[["year", "day"]].apply(make_full_date, axis=1)
        self.target_df["day_target_categorical"] = self.target_df["hydro_fixed_day_categorical"]

        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.target_cols = target_cols

        self.gap = gap
        self.history = history

    def __len__(self):
        return self.target_df.shape[0]

    def __getitem__(self, idx):
        station = self.target_df["hydro_station_id"].values[idx]

        # get targets
        target_day = self.target_df.full_date.values[idx]
        target_day_feature = self.target_df.day_target_categorical.iloc[idx]
        target = self.target_df[self.target_cols].values[idx]

        end_day = target_day - timedelta(days=self.gap)
        start_day = end_day - timedelta(days=self.history - 1)

        features_mask = (
            (self.full_df["full_date"] >= start_day)
            & (self.full_df["full_date"] <= end_day)
            & (self.full_df["hydro_station_id"] == station)
        )

        feat_matrix = self.full_df[features_mask][self.cat_cols + self.num_cols].values
        new_feat_matrix = np.zeros((self.history, feat_matrix.shape[1]))

        if features_mask.sum() == 0:
            return {"x": new_feat_matrix, "y": target, "day": target_day_feature, "station": 0}

        new_feat_matrix[-feat_matrix.shape[0] :] = feat_matrix
        encoded_station_id = self.full_df[features_mask]["hydro_fixed_station_id_categorical"].values[0]

        return {"x": new_feat_matrix, "y": target, "day": target_day_feature, "station": encoded_station_id}


class LenaDataModule(pl.LightningDataModule):
    def __init__(self, train, test, features_df, batch_size=128, num_workers=2, train_only=True):
        super().__init__()

        self.train = train
        self.test = test
        self.features_df = features_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_only = train_only

        self.numerical_features = [c for c in features_df if "numeric" in c and "namask" not in c]
        self.categorical_features = [c for c in features_df if "categorical" in c and "namask" not in c]

        if train_only:
            train_list.extend(val_list)

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
        self.test_ds = LenaDataset(
            self.test.loc[self.test.year.isin(test_list)],
            self.features_df,
            self.categorical_features,
            self.numerical_features,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        train_sampler = RandomSampler(self.train_ds)
        return DataLoader(
            self.train_ds, sampler=train_sampler, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        valid_sampler = SequentialSampler(self.valid_ds)
        return DataLoader(
            self.valid_ds, sampler=valid_sampler, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        test_sampler = SequentialSampler(self.test_ds)
        return DataLoader(self.test_ds, sampler=test_sampler, batch_size=self.batch_size, num_workers=self.num_workers)


class LenaDataModuleExtra(pl.LightningDataModule):
    def __init__(self, features_df, gap=66, history=139, batch_size=128, num_workers=2):
        super().__init__()

        self.features_df = features_df
        self.gap = gap
        self.history = history

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.numerical_features = [c for c in features_df if "numeric" in c and "namask" not in c]
        self.categorical_features = [c for c in features_df if "categorical" in c and "namask" not in c]
        self.target_cols = [c for c in features_df if "water_code_" in c and "namask" not in c]

        self.train_ds = LenaDatasetExtra(
            full_df=self.features_df.loc[self.features_df.year.isin(train_list)].reset_index(drop=True).copy(),
            cat_cols=self.categorical_features,
            num_cols=self.numerical_features,
            target_cols=self.target_cols,
            gap=self.gap,
            history=self.history,
            mode="train",
        )
        self.valid_ds = LenaDatasetExtra(
            full_df=self.features_df.loc[self.features_df.year.isin(val_list)].reset_index(drop=True).copy(),
            cat_cols=self.categorical_features,
            num_cols=self.numerical_features,
            target_cols=self.target_cols,
            gap=self.gap,
            history=self.history,
            mode="val",
        )
        self.test_ds = LenaDatasetExtra(
            full_df=self.features_df.loc[self.features_df.year.isin(test_list)].reset_index(drop=True).copy(),
            cat_cols=self.categorical_features,
            num_cols=self.numerical_features,
            target_cols=self.target_cols,
            gap=self.gap,
            history=self.history,
            mode="test",
        )

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        train_sampler = RandomSampler(self.train_ds)
        return DataLoader(
            self.train_ds, sampler=train_sampler, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        valid_sampler = SequentialSampler(self.valid_ds)
        return DataLoader(
            self.valid_ds, sampler=valid_sampler, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        test_sampler = SequentialSampler(self.test_ds)
        return DataLoader(self.test_ds, sampler=test_sampler, batch_size=self.batch_size, num_workers=self.num_workers)
