from vae_ts_test.models import RNNEncoder, RNNDecoder, LinearDecoder, LinearEncoder
import torch
from torch import nn
import pytorch_lightning as pl
import constants as const
pl.seed_everything(1234)


class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=4, learning_rate=1e-3,
                 latent_dim=4, input_size=2,
                 rnn_layers=5,
                 seq_len=100, beta=10, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = RNNEncoder(input_size=input_size, hidden_size=enc_out_dim, num_layers=rnn_layers)
        self.decoder = RNNDecoder(input_size=latent_dim, num_layers=rnn_layers,
                               output_size=input_size, seq_len=seq_len)

        # self.encoder = LinearEncoder(**const.HPARAMS)
        # self.decoder = LinearDecoder(**const.HPARAMS)

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale_diag = nn.Parameter(torch.zeros(seq_len*input_size))

        # for beta term of beta-variational autoencoder
        self.beta = beta
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def gaussian_likelihood(self, x_hat, logscale_diag, x):
        scale_diag = torch.exp(logscale_diag)
        scale = torch.diag(scale_diag)
        mu_x = x_hat.reshape(x_hat.shape[0], -1)
        dist = torch.distributions.MultivariateNormal(mu_x, scale_tril=scale)

        # measure prob of seeing x under p(x|z)
        log_pxz = dist.log_prob(x.reshape(x.shape[0], -1))
        # return log_pxz.sum(dim=(1, 2))
        return log_pxz

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

    def forward(self, x):
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        return mu, std, z, x_hat, self.log_scale_diag


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
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale_diag, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (self.beta * kl - recon_loss)
        elbo = elbo.mean()

        log_dict = dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
        })

        return elbo, log_dict

    def training_step(self, batch, batch_idx):
        x, _ = batch
        train_elbo, train_log_dict = self._shared_eval(x)
        self.logger.experiment.add_scalars("loss", dict(
            elbo_train=train_log_dict['elbo'],
            kl_train=train_log_dict['kl'],
            recon_loss_train=train_log_dict['recon_loss']
        ))
        return train_elbo

    def validation_step(self, batch, batch_idx):
        x, _ = batch
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
    from vae_ts_test.callbacks import PlottingCallBack
    import constants as const
    # resume
    # LAST_CKP = 'lightning_logs/version_8/checkpoints/epoch=999-step=8999.ckpt'
    # trainer = pl.Trainer(resume_from_checkpoint=LAST_CKP, max_epochs=10000)
    # vae = VAE.load_from_checkpoint(LAST_CKP, **const.HPARAMS)

    # start new
    pl.seed_everything(123)
    trainer = pl.Trainer(callbacks=[PlottingCallBack()], gpus=1, max_epochs=None,
                         log_every_n_steps=const.HPARAMS['log_every_n_steps'])
    vae = VAE(**const.HPARAMS)
    dm = RandomCurveDataModule(**const.HPARAMS)
    trainer.fit(vae, dm)


if __name__ == '__main__':
    train()
