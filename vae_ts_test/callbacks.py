from pytorch_lightning.callbacks import Callback
from vae_ts_test.dataset import SimpleRandomCurvesDataset
import constants as const
from torch.utils.data import DataLoader
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import os
import torch
import numpy as np


class PlottingCallBack(Callback):
    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch in [i*10 for i in range(10000000)]:
            epoch = trainer.current_epoch
            version_str = trainer.log_dir.split('/')[-1]
            with torch.no_grad():
                x_batch, idxs, mu_z, std_z, z_sample, mu_x, std_x = self.get_data(
                    pl_module)
                data = (x_batch, idxs, mu_z, std_z, z_sample, mu_x, std_x)
                self.save_scatter_fig(
                    epoch=epoch,
                    version_str=version_str,
                    pl_module=pl_module,
                    data=data)
                self.save_recon_plot(
                    epoch=epoch,
                    version_str=version_str,
                    pl_module=pl_module,
                    data=data,
                    index=0)
                self.save_z_dist_plot(
                    epoch=epoch,
                    version_str=version_str,
                    pl_module=pl_module,
                    data=data)

    @staticmethod
    def get_data(pl_module):
        dataset = SimpleRandomCurvesDataset(
            const.DATA_PATH, const.HIDDEN_STATE_PATH)
        dataset.df_data.head()
        batch_size = 100
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=24)
        batches = iter(dataloader)
        x_batch, idxs = batches.next()
        mu_z, std_z, z_sample, mu_x, std_x = pl_module.forward(x_batch.cuda())
        return x_batch, idxs, mu_z, std_z, z_sample, mu_x, std_x

    @staticmethod
    def save_recon_plot(epoch, version_str, pl_module, data, index):
        x_batch, idxs, mu_z, std_z, z_sample, mu_x, std_x = data
        x = list(range(const.TIMESTEPS))
        x_sensor_1 = x_batch.detach().numpy()[index, :, 0]
        x_sensor_2 = x_batch.detach().numpy()[index, :, 1]
        mu_rec_sensor_1 = mu_x.detach().cpu().numpy()[index, :, 0]
        mu_rec_sensor_2 = mu_x.detach().cpu().numpy()[index, :, 1]
        log_scale = pl_module.log_scale_diag.detach().cpu().numpy()
        std = np.exp(log_scale)
        std = std.reshape(-1, 2)
        sensor_1_upper = list(mu_rec_sensor_1 + 2*std[:, 0])
        sensor_2_upper = list(mu_rec_sensor_2 + 2*std[:, 1])
        sensor_1_lower = list(mu_rec_sensor_1 - 2*std[:, 0])
        sensor_2_lower = list(mu_rec_sensor_2 - 2*std[:, 1])

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        # signal 1
        for sig, name, colour in zip([x_sensor_1, mu_rec_sensor_1],
                                     ['x_s1', 'p(x_s2|z)'], ['rgb(0,0,100)', 'rgba(192,58,58)']):
            fig.add_trace(
                go.Scatter(x=x,
                           y=sig, name=name,
                           line=dict(color=colour),
                           mode="lines", opacity=.5),
                row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x+x[::-1],  # x, then x reversed
            # upper, then lower reversed
            y=sensor_1_upper + sensor_1_lower[::-1],
            fill='toself',
            fillcolor='rgba(192,58,58,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False), row=1, col=1)
        fig.update_yaxes(range=[-2, 2])

        # signal 2
        for sig, name, colour in zip([x_sensor_2, mu_rec_sensor_2],
                                     ['x_s2', 'p(x_s2|z)'], ['rgb(0,0,100)', 'rgba(192,58,58)']):
            fig.add_trace(
                go.Scatter(x=x,
                           y=sig, name=name,
                           line=dict(color=colour),
                           mode="lines", opacity=.5),
                row=2, col=1,
            )

        fig.add_trace(go.Scatter(
            x=x+x[::-1],  # x, then x reversed
            # upper, then lower reversed
            y=sensor_2_upper + sensor_2_lower[::-1],
            fill='toself',
            fillcolor='rgba(192,58,58,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True
        ),
            row=2, col=1,
        )
        fig.update_yaxes(range=[-2, 2])
        fig.update_xaxes(title='time')

        fig.update_layout(height=1000, width=1200, title_text=f"Example sample for x and p(x|z), epoch: {epoch}",
                          showlegend=True)
        if not os.path.exists(f"images/{version_str}"):
            os.mkdir(f"./images/{version_str}")

        fig.write_image(
            f'./images/{version_str}/recon_plot_{epoch:05d}.png')

    @staticmethod
    def save_scatter_fig(epoch, version_str, pl_module, data):
        x_batch, idxs, mu_z, std_z, z_sample, mu_x, std_x = data

        df_hidden_states = pd.read_csv(const.HIDDEN_STATE_PATH)
        df_hidden_states[df_hidden_states.index.isin(idxs.detach().numpy())]
        df_latent_mu = pd.DataFrame(mu_z.cpu().detach().numpy(),
                                    columns=[f'mu_{i}' for i in range(const.HPARAMS['latent_dim'])])

        fig = make_subplots(rows=4, cols=4)

        for i, hs in enumerate(df_hidden_states.columns):
            for j, hs_pred in enumerate(df_latent_mu.columns):
                fig.add_trace(go.Scatter(
                    x=df_hidden_states[hs], y=df_latent_mu[hs_pred],
                    mode='markers', name=f'activation {hs_pred} over box_x',
                    marker_color='#1f77b4'),
                    row=i+1, col=j+1)
                fig.update_yaxes(range=[-2, 2])

        # # Update xaxis properties
        for i in range(const.HPARAMS['latent_dim']):
            fig.update_xaxes(
                title_text=df_hidden_states.columns[0], row=1, col=i+1)
            fig.update_xaxes(
                title_text=df_hidden_states.columns[1], row=2, col=i+1)
            fig.update_xaxes(
                title_text=df_hidden_states.columns[2], row=3, col=i+1)
            fig.update_xaxes(
                title_text=df_hidden_states.columns[3], row=4, col=i+1)

        # Update xaxis properties
        for j in range(len(df_hidden_states)):
            fig.update_yaxes(
                title_text=df_latent_mu.columns[0], row=j+1, col=1)
            fig.update_yaxes(
                title_text=df_latent_mu.columns[1], row=j+1, col=2)
            fig.update_yaxes(
                title_text=df_latent_mu.columns[2], row=j+1, col=3)
            fig.update_yaxes(
                title_text=df_latent_mu.columns[3], row=j+1, col=4)


        fig.update_layout(height=1000, width=1600,
                          title_text=f"Latent neuron activations vs. hidden states, epoch: {epoch}",
                          showlegend=False)

        if not os.path.exists(f"images/{version_str}"):
            os.mkdir(f"./images/{version_str}")

        fig.write_image(
            f'./images/{version_str}/scatter_lat_vars_epoch_{epoch:05d}.png')

    @staticmethod
    def save_z_dist_plot(epoch, version_str, pl_module, data):
        x_batch, idxs, mu_z, std_z, z_sample, mu_x, std_x = data
        df_latent_mu = pd.DataFrame(mu_z.detach().cpu().numpy(),
                                    columns=[f'mu_{i}' for i in range(const.HPARAMS['latent_dim'])])
        df_latent_std = pd.DataFrame(std_z.detach().cpu().numpy(), columns=[
                                     f'std_{i}' for i in range(const.HPARAMS['latent_dim'])])
        fig = make_subplots(rows=1, cols=2)
        for col in df_latent_mu.columns:
            fig.add_trace(go.Histogram(
                x=df_latent_mu[col], name=col), row=1, col=1)

        for col in df_latent_std.columns:
            fig.add_trace(go.Histogram(
                x=df_latent_std[col], name=col), row=1, col=2)

        # Overlay both histograms
        fig.update_layout(barmode='overlay')
        fig.update_layout(title_text=f"Distribution of distribution parameters for z (Gaussian mu and std), epoch: {epoch} ",
                          showlegend=True)
        fig.update_xaxes(range=[-5, +5], title_text='mu', row=1, col=1)
        fig.update_xaxes(range=[0, 1.5],title_text='std', row=1, col=2)
        for row in (1, 2):
            fig.update_yaxes(title_text='frequency', row=row, col=1)

        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.4)

        if not os.path.exists(f"images/{version_str}"):
            os.mkdir(f"./images/{version_str}")

        fig.write_image(
            f'./images/{version_str}/lat_var_dist_epoch_{epoch:05d}.png')
