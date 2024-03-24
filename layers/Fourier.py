import numpy as np
import torch
import torch.nn as nn

def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


# ########## fourier layer #############
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        # get modes on frequency domain
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        # print('modes={}, index len={}, index={}'.format(modes, len(self.index), self.index))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel ), (in_channel, out_channel) -> (batch, out_channel)
        return torch.einsum("bh,hi->bi", input, weights)

    def forward(self, q):
        B, L, E = q.shape
        x = q.permute(0, 2, 1)   # [B, L, E] --> [B, E, L]
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        # print('FourierBlock x_ft: ', x_ft.shape)
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        # print('FourierBlock out_ft: ', out_ft.shape)
        ## Random
        for wi, i in enumerate(self.index):
            out_ft[:, :, wi] = self.compl_mul1d(x_ft[:, :, i], self.weights1[:, :, wi])
        # ## Top k
        # k_l = len(self.index)-1
        # for i in range(len(self.index)):
        #     out_ft[:, :, i] = self.compl_mul1d(x_ft[:, :, k_l-i], self.weights1[:, :, i])

        # Return to time domain
        # print('FourierBlock q: ', q.shape)
        # print('FourierBlock x: ', x.shape)
        # print('FourierBlock self.weights1: ', self.weights1.shape)
        # print('FourierBlock out_ft: ', out_ft.shape)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        x = x.permute(0, 2, 1)   # [B, E, L] --> [B, L, E]
        return (x, None)

