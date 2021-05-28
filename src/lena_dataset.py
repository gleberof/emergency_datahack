import numpy as np
from torch.utils.data import Dataset


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
        feat_matrix = self.full_df.loc[(part_index1 | part_index2) & part_index3, self.cat_cols + self.num_cols].values
        new_feat_matrix = np.zeros((139, feat_matrix.shape[0]))
        new_feat_matrix[-feat_matrix.shape[0] :] = feat_matrix
        return {"x": new_feat_matrix, "y": target, "day": day}
