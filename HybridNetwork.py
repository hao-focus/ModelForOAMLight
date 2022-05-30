# Author Hao Wang & Ziyu Zhan
# OAM spectrum detection based on hybrid optoeletronic neural network
#  ============================
import torch
import numpy as np
import torch.fft as fourier
import torch.nn as nn
import math
import os
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# ================== global parameters============================
size = 200
batch_size = 2
distance = 40
distance_pro = 40
distance_det = 60
ls = 106.7
fs = 1.0 / (2 * ls)
wl = 1
k = 2.0 * math.pi / wl
paddings = 100
kernel_size = 4
stride = 4
shift = 50
scale = 11.0
extension = 2.0
hidden_1 = 2500
T = 0.1
dim_in = 2500
dim_out = 21
# Ic = torch.abs(0.01 * torch.randn(200, 200, 200))
# Ic = torch.zeros(200, 200, 200)
# Ic[:, 150: 200, 150: 200] = 1
# Ic = Ic.cuda()
ph = np.fromfunction(
    lambda x, y: ((x - (size + 2 * paddings) // 2) * fs) ** 2 + ((y - (size + 2 * paddings) // 2) * fs) ** 2,
    shape=(size + 2 * paddings, size + 2 * paddings), dtype=np.complex64)
h = np.fft.fftshift(np.exp(1.0j * distance * np.sqrt(k ** 2 - np.multiply(4 * np.pi ** 2, ph))))
h = torch.from_numpy(h)
h = h.cuda()
h_pro = np.fft.fftshift(np.exp(1.0j * distance_pro * np.sqrt(k ** 2 - np.multiply(4 * np.pi ** 2, ph))))
h_pro = torch.from_numpy(h_pro)
h_pro = h_pro.cuda()
h_det = np.fft.fftshift(np.exp(1.0j * distance_det * np.sqrt(k ** 2 - np.multiply(4 * np.pi ** 2, ph))))
h_det = torch.from_numpy(h_det)
h_det = h_det.cuda()

def diff(wave, trans):
    wave = F.pad(wave, pad=[paddings, paddings, paddings, paddings])
    wave = torch.squeeze(wave)
    wave_f = fourier.fft2(fourier.fftshift(wave))
    wave_f *= trans
    wave = fourier.ifftshift(fourier.ifft2(wave_f))
    return wave

def modulation(wave, phase):
    wave_m = wave * torch.exp(1.0j * scale * math.pi * (torch.sin(extension * phase) + 1))
    wave_mf = wave_m[:, paddings: paddings + size, paddings: paddings + size]
    return wave_mf

class Net(torch.nn.Module):
    def __init__(self, num_layers):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.phase = torch.nn.Parameter(np.sqrt(0.0 * math.pi) * torch.randn(num_layers, size, size), requires_grad=True)  # Todo: check the initialization
        self.avg_pool = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, stride=stride))
        self.fcl = nn.Sequential(
            nn.Linear(dim_in, hidden_1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_1, dim_out),
        )
        self.BN = nn.Sequential(nn.BatchNorm2d(1))
        self.sm = nn.Sequential(nn.Softmax(dim=1))

    def forward(self, x):
        x = diff(x, h_pro)
        x = x[:, paddings: paddings + size, paddings: paddings + size]
        for idx in range(self.num_layers):
            trans = F.pad(self.phase[idx], pad=[paddings, paddings, paddings, paddings])
            trans = trans.cuda()
            x = modulation(diff(x, h), trans)
        x = diff(x, h_det)
        x = x[:, paddings: paddings + size, paddings: paddings + size]
        i_output = torch.square(torch.abs(x))
        # i_output = i_output * Ic
        ccd_signal = self.avg_pool(i_output)
        ccd_signal = torch.unsqueeze(ccd_signal, dim=1).float()
        ccd_signal = self.BN(ccd_signal)
        ccd_signal = torch.squeeze(ccd_signal)
        ccd_signal = torch.flatten(ccd_signal, start_dim=1, end_dim=-1).float()
        ccd_signal = self.fcl(ccd_signal)
        ccd_signal = ccd_signal / T
        ccd_signal = self.sm(ccd_signal)
        return ccd_signal
