
from vae_ts_test.models import LinearDecoder, LinearEncoder, RNNDecoder, RNNEncoder
import torch
from torch import nn
import pytorch_lightning as pl
pl.seed_everything(1234)




def train():
    # parser = ArgumentParser()
    # parser.add_argument('--gpus', type=int, default=None)
    # parser.add_argument('--dataset', type=str, default='cifar10')
    # args = parser.parse_args()
    from vae_ts_test.data_module import RandomCurveDataModule
    from vae_ts_test.callbacks import PlottingCallBack, BetaIncreaseCallBack
    import constants as const
    # resume
    # LAST_CKP = 'lightning_logs/version_8/checkpoints/epoch=999-step=8999.ckpt'
    # trainer = pl.Trainer(resume_from_checkpoint=LAST_CKP, max_epochs=10000)
    # vae = VAE.load_from_checkpoint(LAST_CKP, **const.HPARAMS)

    # start new
    trainer = pl.Trainer(callbacks=[PlottingCallBack(), BetaIncreaseCallBack()],
                         gpus=1, max_epochs=100_000,
                         log_every_n_steps=const.HPARAMS['log_every_n_steps'])
    vae = VAE(**const.HPARAMS)
    dm = RandomCurveDataModule(**const.HPARAMS)
    trainer.fit(vae, dm)


if __name__ == '__main__':
    train()
