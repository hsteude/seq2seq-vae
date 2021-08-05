# Seq2Seq VAE for time series representation learning

This is just a repo to play around with VAEs. In this case I'm curious if a VAE
is able to find things like phase shifts, amplitudes and frequencies in a couple
of sin wave-like signals.

## Concrete Scenario
We have two time series. Both sin signals ($s_1 = a_1sin(2\pi f)$ and $s_2 =
a_2sin(2\pi f + \phi)$). To generate the data we assume $a_1, a_2, f, and \phi$
to be uniform random variables in some interval.
Now we use a BetaVAE and try to encode these four hidden states into the latend
space.

## Installation 
Clone this repo, go to the project root and run (I'm assuming you run some unix
system are using conda):
```
conda create -n vae-ts python=3.9
conda activate vae-ts
pip install -e . 
```

## Run this thing

You'll first need to create the data set using this notebook: 
./notebooks/01_datagen.ipynb.
Now you can go a head an start a training job like this:
```shell
ipython vae_ts_test.vae.py
```
Note that the hyper parameters and some other constants are stored in the file
./constants.py. Here you might also adjust if you want to use GPUs and stuff.







