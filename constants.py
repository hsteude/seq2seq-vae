DATA_PATH = './data/simple_random_curves.csv'
HIDDEN_STATE_PATH = './data/hidden_states.csv'
TIMESTEPS = 50
SAMPLES = 10000
FREQ_MIN = .01
FREQ_MAX = .05
AMP_MAX = 1.5
AMP_MIN = .5
PHI_MAX = 1/2
PHI_MIN = 0
SEED = 123


HPARAMS = dict(enc_out_dim=100,  # needs to be the same as hidden_size
               hidden_size=100,
               rnn_layers=1,
               latent_dim=5,
               input_size=2,
               seq_len=TIMESTEPS,
               validdation_split=.1,
               batch_size=1000,
               dl_num_workers=20,
               beta=0.5,  # initial value, will be changed through callback during training
               learning_rate=5e-3,
               log_every_n_steps=10,
               num_epochs_constant_beta=200,
               beta_max=10,
               beta_increase_steps=20,
               rnn=True)
