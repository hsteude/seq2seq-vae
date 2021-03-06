import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from vae_ts_test.dataset import SimpleRandomCurvesDataset
import constants as const

import numpy as np
import torch
from torch.utils.data import DataLoader


class RandomCurveDataModule(pl.LightningDataModule):
    def __init__(self, validation_split, batch_size, dl_num_workers,
                 *args, **kwargs):
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.dl_num_workers = dl_num_workers
        super().__init__()

    def setup(self, stage=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        # data set and loader related
        dataset_full = SimpleRandomCurvesDataset(data_path=const.DATA_PATH,
                                                 hidden_states_path=const.HIDDEN_STATE_PATH)
        dataset_size = len(dataset_full)
        len_val = int(np.floor(dataset_size * self.validation_split))
        len_train = dataset_size - len_val
        self.dataset_train, self.dataset_val = random_split(
            dataset=dataset_full, lengths=[len_train, len_val],
            generator=torch.Generator())

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          num_workers=self.dl_num_workers,
                          pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=self.batch_size,
                          num_workers=self.dl_num_workers,
                          pin_memory=False)


if __name__ == '__main__':
    # quick and dirty test
    hparams = dict(
        validation_split=.1,
        batch_size=10,
        dl_num_workers=6
    )
    rcdm = RandomCurveDataModule(**hparams)
    rcdm.setup()
    dl = rcdm.train_dataloader()
    ts_train_batch = iter(dl).next()

