import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Fourier import FourierBlock


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    : Trend  & Seasonal 추출
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]

        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean

        return res, moving_mean 


class EncoderLayer(nn.Module):
    def __init__(self, configs):
        super(EncoderLayer, self).__init__()

        self.fft_block = FourierBlock(in_channels=configs.enc_in,
                                      out_channels=configs.enc_in,
                                      seq_len=configs.seq_len,
                                      modes=configs.modes,
                                      mode_select_method=configs.mode_select)

        self.conv1 = nn.Conv1d(in_channels=configs.enc_in, out_channels=configs.d_model, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.enc_in, kernel_size=1, bias=False)

        self.decomp1 = series_decomp_multi(configs.trend_kernels)
        self.decomp2 = series_decomp_multi(configs.trend_kernels)

        self.dropout = nn.Dropout(configs.dropout)
        self.activation = F.relu if configs.activation == "relu" else F.gelu

    def forward(self, x):

        # FFT를 이용하여 변동성이 감소된 데이터 추출
        new_x, attn = self.fft_block(x)

        # trend를 제외시켜 더 강한 High frequency 추출
        x, _ = self.decomp1(new_x)
        y = x

        # Feed Forward
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # trend를 제외시켜 더 강한 High frequency 추출
        res, _ = self.decomp2(x + y)

        return res, attn


class Encoder(nn.Module):
    """
    노이즈성이 아닌 이벤트, 변환점의 중요 정보를 갖는 Global High Frequency 추출 진행
    """
    def __init__(self, configs, norm_layer=None):
        super(Encoder, self).__init__()

        attn_layers = [EncoderLayer(configs) for l in range(configs.e_layers)]

        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Decoder_Lstm(nn.Module):
    def __init__(self, seq_len, c_in, c_out, device):
        super(Decoder_Lstm, self).__init__()
        self.lstm1 = nn.LSTM(c_in, 32, batch_first=True)
        self.lstm2 = nn.LSTM(32, 64, batch_first=True)
        self.lstm3 = nn.LSTM(64, 64, batch_first=True)
        self.head = nn.Linear(seq_len*64, c_out)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        outputs, _ = self.lstm1(x)
        outputs, _ = self.lstm2(outputs)
        outputs, _ = self.lstm3(outputs)
        outputs = self.head(outputs.reshape(outputs.size(0), -1))
        return self.sigmoid(outputs)

