import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import constants as const


class SimpleRandomCurvesDataset(Dataset):
    """Write me!"""

    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.df_scaled = self.scale_ts(df)

    @staticmethod
    def scale_single_ts(ts, mean, std):
        return (ts - mean) / std

    def scale_ts(self, df):
        df_scaled = pd.DataFrame(
            np.array([self.scale_single_ts(df.iloc[i, :], df.iloc[:, -1].mean(), df.iloc[:, -1].std())
                      for i in range(len(df))]))
        return df_scaled

    def __len__(self):
        """Size of dataset
        """
        return len(self.df_scaled)

    def __getitem__(self, index):
        """Get one sample (including questions and answers)"""
        return self.df_scaled.iloc[index, :].values


if __name__ == '__main__':
    # test for lets have a look
    dataset = SimpleRandomCurvesDataset(csv_file=const.DATA_PATH)
    idx = 10
    ts = dataset[idx]
    print(type(ts))
