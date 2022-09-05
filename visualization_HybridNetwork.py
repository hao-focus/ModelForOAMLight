# Hybrid OAM spectrum DL network
import torch
import numpy as np
import torch.fft as fourier
import torch.nn as nn
import math
import os
import torch.nn.functional as F
import visual_para as para
import scipy.sparse as sp
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# free space propagation
ph = np.fromfunction(
    lambda x, y: ((x - (para.size + 2 * para.paddings) // 2) * para.fs) ** 2 + ((y - (para.size + 2 * para.paddings) // 2) * para.fs) ** 2,
    shape=(para.size + 2 * para.paddings, para.size + 2 * para.paddings), dtype=np.complex64)
h = np.fft.fftshift(np.exp(1.0j * para.distance * np.sqrt(para.k ** 2 - np.multiply(4 * np.pi ** 2, ph))))
h = torch.from_numpy(h)
h = h.cuda()
h_pro = np.fft.fftshift(np.exp(1.0j * para.distance_pro * np.sqrt(para.k ** 2 - np.multiply(4 * np.pi ** 2, ph))))
h_pro = torch.from_numpy(h_pro)
h_pro = h_pro.cuda()
h_det = np.fft.fftshift(np.exp(1.0j * para.distance_det * np.sqrt(para.k ** 2 - np.multiply(4 * np.pi ** 2, ph))))
h_det = torch.from_numpy(h_det)
h_det = h_det.cuda()
c1 = torch.arange(-10, 11, 1)

def diff(wave, trans):
    wave = F.pad(wave, pad=[para.paddings, para.paddings, para.paddings, para.paddings])
    wave = torch.squeeze(wave)
    wave_f = fourier.fft2(fourier.fftshift(wave))
    wave_f *= trans
    wave = fourier.ifftshift(fourier.ifft2(wave_f))
    return wave

def modulation(wave, phase):
    wave_m = wave * torch.exp(1.0j * para.scale * math.pi * (torch.sin(para.extension * phase) + 1))
    wave_mf = wave_m[:, para.paddings: para.paddings + para.size, para.paddings: para.paddings + para.size]
    return wave_mf

def bar_show(img, x):
    img_np = img.numpy()
    plt.bar(x, img_np, color='green')
    plt.show()

class Net(torch.nn.Module):
    def __init__(self, num_layers):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.phase = torch.nn.Parameter(np.sqrt(0.0 * math.pi) * torch.randn(num_layers, para.size, para.size), requires_grad=True)  # Todo: check the initialization
        self.avg_pool = nn.Sequential(nn.AvgPool2d(kernel_size=para.kernel_size, stride=para.stride))
        self.fcl = nn.Sequential(
            nn.Linear(para.dim_in, para.hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(para.hidden, para.dim_out),
        )
        self.BN = nn.Sequential(nn.BatchNorm2d(1))
        self.sm = nn.Sequential(nn.Softmax(dim=1))

    def forward(self, x, batch_size):
        result = torch.zeros(para.map_size, para.map_size)
        ch_map = torch.zeros(para.map_size, para.map_size)
        temp = torch.zeros(para.range)
        delta = torch.zeros(batch_size)
        delta_max = torch.zeros(batch_size)
        histo = torch.zeros(para.range + 1)
        x1 = diff(x, h_pro)
        x = x1[:, para.paddings: para.paddings + para.size, para.paddings: para.paddings + para.size]
        for idx in range(self.num_layers):
            trans = F.pad(self.phase[idx], pad=[para.paddings, para.paddings, para.paddings, para.paddings])
            trans = trans.cuda()
            x = modulation(diff(x, h), trans)
        x = diff(x, h_det)
        x = x[:, para.paddings: para.paddings + para.size, para.paddings: para.paddings + para.size]
        i_output = torch.square(torch.abs(x))
        for i in range(para.map_size):
            for j in range(para.map_size):
                I_block = torch.zeros(batch_size, para.map_size, para.map_size)
                I_block[:, i: i + para.win_size, j: j + para.win_size] = 1
                I_block = I_block.cuda()
                ccd_signal = self.avg_pool(i_output)
                ccd_signal = ccd_signal * I_block
                ccd_signal = torch.unsqueeze(ccd_signal, dim=1).float()
                ccd_signal = self.BN(ccd_signal)
                ccd_signal = torch.squeeze(ccd_signal)
                ccd_signal = torch.flatten(ccd_signal, start_dim=1, end_dim=-1).float()
                ccd_signal = self.fcl(ccd_signal)  # b 81
                ccd_signal = ccd_signal / para.T
                ccd_signal = self.sm(ccd_signal)
                for kk in range(batch_size):
                    a = torch.argmax(ccd_signal[kk, :])
                    a2 = torch.argmin(ccd_signal[kk, :])
                    b = ccd_signal[kk, a]
                    b2 = ccd_signal[kk, a2]
                    delta[kk] = b - b2
                    delta_max[kk] = a
                    temp[a] += 1
                final = torch.argmax(temp)
                for xx in range(batch_size):
                    if delta_max[xx] == final:
                        delta_max[xx] = 1
                    else:
                        delta_max[xx] = 0
                delta = delta * delta_max
                AS = sp.csr_matrix(delta)
                delta = torch.tensor(AS.data)
                delta1 = torch.mean(delta)
                temp = torch.zeros(para.range + 1)
                final = final + 1
                final = final.cuda()
                if delta1 < para.thershold:
                    final = 0
                print(final)
                if final == 0:
                    result[i, j] = final
                else:
                    result[i, j] = final + para.shift
                histo[final] += 1
                if final != 0:
                    ch_map[i, j] = 1
                delta = torch.zeros(batch_size)
                delta_max = torch.zeros(batch_size)
        return result, ch_map, histo
