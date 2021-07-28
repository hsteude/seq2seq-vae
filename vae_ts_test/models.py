from torch import nn
import torch

class Encoder(nn.Module):
    """Please implement me"""
    def __init__(self, input_size: int = 1,
                 hidden_size: int = 4,
                 *args, **kwargs):
        super(Encoder, self).__init__()
        self.layer1 = nn.LSTM(input_size=input_size, batch_first=True,
                              hidden_size=hidden_size).float()
        self.hidden_size = hidden_size

    def forward(self, x):
        # x.shape is (batch, seq_len, input_size)
        x, (final_state, _) = self.layer1(x)
        # the embedding shape is (seq_len[in this case 1], batch, input_size)
        # we return in the shape we are used to (batch, seq_len, input_size)
        return final_state.reshape(-1, 1, self.hidden_size)


class Decoder(nn.Module):
    """Please write me"""
    def __init__(self,
                 input_size=4,
                 output_size=1,
                 seq_len=100, *args, **kwargs):
        super(Decoder, self).__init__()
        self.layer1 = nn.LSTM(input_size=input_size, batch_first=True,
                              hidden_size=output_size).float()
        self.seq_len = seq_len

    def forward(self, x):
        # so x has the shape (batch, seq_lwn, input_size)
        x = x.repeat(1, self.seq_len, 1)
        x, (_, _) = self.layer1(x)
        return x

