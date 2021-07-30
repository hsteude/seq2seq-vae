from vae_ts_test.models import Encoder, Decoder
from argparse import ArgumentParser
import torch
from torch import nn
import pytorch_lightning as pl
pl.seed_everything(1234)


class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=4,
                 latent_dim=4, input_size=1,
                 seq_len=100, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = Encoder(input_size=input_size, hidden_size=enc_out_dim)
        self.decoder = Decoder(input_size=latent_dim,
                               output_size=input_size, seq_len=seq_len)

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing x under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def _shared_eval(self, x):

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        log_dict = dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
        })

        return elbo, log_dict

    def training_step(self, batch, batch_idx):
        x = batch
        train_elbo, train_log_dict = self._shared_eval(x)
        self.logger.experiment.add_scalars("loss", dict(
            elbo_train=train_log_dict['elbo'],
            kl_train=train_log_dict['kl'],
            recon_loss_train=train_log_dict['recon_loss']
        ))
        return train_elbo

    def validation_step(self, batch, batch_idx):
        x = batch
        self._shared_eval(x)
        val_elbo, val_log_dict = self._shared_eval(x)
        self.logger.experiment.add_scalars("loss", dict(
            elbo_val=val_log_dict['elbo'],
            kl_val=val_log_dict['kl'],
            recon_loss_val=val_log_dict['recon_loss']
        ))
        return val_elbo


def train():
    # parser = ArgumentParser()
    # parser.add_argument('--gpus', type=int, default=None)
    # parser.add_argument('--dataset', type=str, default='cifar10')
    # args = parser.parse_args()
    from vae_ts_test.data_module import RandomCurveDataModule

    hparams = dict(enc_out_dim=4, latent_dim=4, input_size=1, seq_len=100,
                   validdation_split=.1,
                   batch_size=10,
                   dl_num_workers=6
                   )
    vae = VAE(**hparams)
    trainer = pl.Trainer(gpus=0, max_epochs=2000)
    dm = RandomCurveDataModule(**hparams)
    trainer.fit(vae, dm)


if __name__ == '__main__':
    train()
