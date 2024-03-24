import torch
import torch.nn as nn
from layers.EncDec import Encoder, my_Layernorm, series_decomp_multi, Decoder_Lstm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    """
    FED-Lstm
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.modes = configs.modes
        self.seq_len = configs.seq_len

        # Decomp
        kernel_size = configs.trend_kernels
        self.decomp = series_decomp_multi(kernel_size)

        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        print('enc_modes: {}, seq len: {}'.format(enc_modes, configs.seq_len))

        self.encoder = Encoder(configs, norm_layer=my_Layernorm(configs.enc_in))
        self.decoder = Decoder_Lstm(configs.seq_len, configs.enc_in, configs.c_out, device)

    def forward(self, x_enc):
        # decomp init
        seasonal_init, trend_init = self.decomp(x_enc)

        # enc: Global High Frequency 추출
        enc_out, attns = self.encoder(x_enc)

        # dec: Global trend + High Frequency, 상승/하락 예측 진행
        enc_trend = trend_init + enc_out
        dec_out = self.decoder(enc_trend)
        return dec_out
