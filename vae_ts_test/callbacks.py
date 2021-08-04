from pytorch_lightning.callbacks import Callback
from vae_ts_test.dataset import SimpleRandomCurvesDataset
import constants as const
from torch.utils.data import DataLoader
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import os
import torch







class PlottingCallBack(Callback):
    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch in [i*10 for i in range(10000000)]:
            epoch = trainer.current_epoch
            version_str = trainer.log_dir.split('/')[-1]
            with torch.no_grad():
                self.save_fig(epoch=epoch, version_str=version_str, pl_module=pl_module)

    @staticmethod
    def save_fig(epoch, version_str, pl_module):
        dataset = SimpleRandomCurvesDataset(const.DATA_PATH, const.HIDDEN_STATE_PATH)
        dataset.df_data.head()
        batch_size = 100
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=24)
        batches = iter(dataloader)
        x_batch, idxs = batches.next()


        mu_z, std_z, z_sample, mu_x, std_x = pl_module.forward(x_batch.cuda())

        df_hidden_states = pd.read_csv(const.HIDDEN_STATE_PATH)
        df_hidden_states[df_hidden_states.index.isin(idxs.detach().numpy())]
        df_latent_mu = pd.DataFrame(mu_z.cpu().detach().numpy().reshape(mu_z.shape[0], mu_z.shape[2]),
                                    columns=[f'mu_{i}' for i in range(const.HPARAMS['latent_dim'])])


        fig = make_subplots(rows=4, cols=4)

        for i, hs in enumerate(df_hidden_states.columns):
            for j, hs_pred in enumerate(df_latent_mu.columns):
                fig.add_trace(go.Scatter(x=df_hidden_states[hs], y=df_latent_mu[hs_pred],
                                    mode='markers', name=f'activation {hs_pred} over box_x',
                                        marker_color='#1f77b4'),
                             row=i+1, col=j+1)
                fig.update_yaxes(range=[-2,2])

        # # Update xaxis properties
        for i in range(const.HPARAMS['latent_dim']):
            fig.update_xaxes(title_text=df_hidden_states.columns[0], row=1, col=i+1)
            fig.update_xaxes(title_text=df_hidden_states.columns[1], row=2, col=i+1)
            fig.update_xaxes(title_text=df_hidden_states.columns[2], row=3, col=i+1)
            fig.update_xaxes(title_text=df_hidden_states.columns[3], row=4, col=i+1)


        for l in range(len(df_hidden_states)):
                fig.update_yaxes(title_text=f"Activation l_{l}", row=l+1, col=1)

        fig.update_layout(height=1000, width=1200, title_text="Latent neuron activations vs. hidden states",
                          showlegend=False)

        if not os.path.exists(f"images/{version_str}"):
            os.mkdir(f"./images/{version_str}")

        fig.write_image(f'./images/{version_str}/scatter_lat_vars_epoch_{epoch:05d}.png')




