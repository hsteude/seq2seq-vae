import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import constants as const


class SimpleRandomCurvesDataset(Dataset):
    """Write me!"""

    def __init__(self, data_path, hidden_states_path):
        self.df_data = pd.read_csv(data_path)
        self.df_hidden_states = pd.read_csv(hidden_states_path)
        self.input_dim = 1

    def __len__(self):
        """Size of dataset
        """
        return len(self.df_hidden_states)

    def __getitem__(self, index):
        """Get one sample (including questions and answers)"""
        out = self.df_data[self.df_data.sample_idx == index][['signal_1', 'signal_2']]\
            .values.astype(np.float32)
        return out, index


if __name__ == '__main__':
    # test for lets have a look
    dataset = SimpleRandomCurvesDataset(data_path=const.DATA_PATH,
                                        hidden_states_path=const.HIDDEN_STATE_PATH)
    idx = 10
    ts = dataset[idx]
    breakpoint()
    print(type(ts))
