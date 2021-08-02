DATA_PATH = './data/simple_random_curves.csv'
TIMESTEPS = 30
SAMPLES = 10000
FREQ_MIN = .025
FREQ_MAX = .05
AMP_MAX = 1.5
AMP_MIN = .5
PHI_MAX = 3.14
PHI_MIN = .0



HPARAMS = dict(enc_out_dim=30,
               latent_dim=3,
               input_size=1,
               seq_len=30,
               validdation_split=.1,
               batch_size=1000,
               dl_num_workers=20,
               beta=5,
               learning_rate=1e-3,
               log_every_n_steps=10
               )
