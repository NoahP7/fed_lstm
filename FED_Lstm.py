import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.EncDec import Encoder, my_Layernorm, series_decomp_multi, Decoder_Lstm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.projection = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.enc_in, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)

        # Decomp
        kernel_size = configs.trend_kernels
        self.decomp = series_decomp_multi(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.dropout)

        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        print('enc_modes: {}, seq len: {}'.format(enc_modes, configs.seq_len))

        self.encoder = Encoder(configs, norm_layer=my_Layernorm(configs.d_model))
        self.decoder = Decoder_Lstm(configs.seq_len, configs.enc_in, configs.c_out, device)

    def forward(self, x_enc, x_mark_enc):
        # decomp init
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out)
        # dec
        enc_cout = self.projection(enc_out.permute(0, 2, 1)).transpose(1, 2)
        enc_trend = trend_init + enc_cout

        dec_out = self.decoder(enc_trend)
        return dec_out
        # print('FEDformer seasonal_part: ', seasonal_part.shape)
        # print('FEDformer trend_part: ', trend_part.shape)
        # final
        # print('FEDformer trend_part: ', trend_part.shape)
        # print('FEDformer seasonal_part: ', seasonal_part.shape)
        # print('FEDformer dec_out: ', dec_out.shape)

        # ## SB
        # dec_out = nn.Sigmoid(dec_out)
        # ##
