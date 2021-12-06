from seq2seq_vae.vae import VAE
from examples.three_tank.data_module import ThreeTankDataModule
import torch
import pytorch_lightning as pl


HPARAMS = dict(
        # datamodule
        validation_split=.1,
        test_split=.05,
        batch_size=100,
        dl_num_workers=6,
        seq_len=50,
        max_epochs=60_000,
        #ValueError
        learning_rate=1e-3,
        latent_dim=5,
        input_size=3,
        beta=1,
        enc_out_dim=100,  # needs to be the same as hidden_size
        hidden_size=100,
        rnn_layers=5,
        rnn_models=True,
    )

def train():
    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=10_000,
        callbacks=[
            # BetaIncreaseCallBack(initial_beta=0, beta_max=1,
                            # number_steps=20, increase_after_n_epochs=200)
            ])
    vae = VAE(**HPARAMS)
    dm = ThreeTankDataModule(**HPARAMS)
    trainer.fit(vae, dm)

if __name__ == '__main__':
    train()
