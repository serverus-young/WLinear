import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FITS(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, configs):
        super(FITS, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.e_in

        self.dominance_freq=configs.cut_freq # 720/24
        self.length_ratio = (self.seq_len + self.pred_len)/self.seq_len
        self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat) # complex layer for frequency upcampling]
        # configs.pred_len=configs.seq_len+configs.pred_len
        # #self.Dlinear=DLinear.Model(configs)
        # configs.pred_len=self.pred_len


    def forward(self, x,x_stamp,y,y_stamp):
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:,self.dominance_freq:]=0 # LPF
        low_specx = low_specx[:,0:self.dominance_freq,:] # LPF
        # print(low_specx.permute(0,2,1))
        low_specxy_ = self.freq_upsampler(low_specx.permute(0,2,1)).permute(0,2,1)
        # print(low_specxy_)
        low_specxy = torch.zeros([low_specxy_.size(0),int((self.seq_len+self.pred_len)/2+1),low_specxy_.size(2)],dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:,0:low_specxy_.size(1),:]=low_specxy_ # zero padding
        low_xy=torch.fft.irfft(low_specxy, dim=1)
        low_xy=low_xy * self.length_ratio # energy compemsation for the length change

        xy=(low_xy) * torch.sqrt(x_var) +x_mean
        return xy