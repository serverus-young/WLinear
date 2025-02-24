import torch
import torch.nn as nn
from pytorch_wavelets import DWT1D, IDWT1D  # 一维离散小波变换
from matplotlib import pyplot as plt
from torch.nn import MSELoss

from utils.metric  import metric
import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WFT(nn.Module):
    def __init__(self, args):
        super(WFT, self).__init__()
        self.configs = args
        # 配置 DWT1D 和 IDWT1D
        self.cwt = DWT1D(wave='db3', mode='zero')  # 一维离散小波正变换
        self.icwt = IDWT1D(wave='db3', mode='zero')  # 一维离散小波逆变换
        self.yl_linear = nn.Linear(args.seq_len//2 + 2, args.d_model//2 + 2)
        self.yh_linear = nn.Linear(args.seq_len//2 + 2, args.d_model//2 + 2)
        self.proj = nn.Linear(args.d_model, args.pred_len )
        # self.wavecore = Waveletcore(args.seq_len//2 + 2, args.pred_len//2 + 2)

    def forward(self, x, x_stamp, y, y_stamp):
        mean = x.mean(dim=1, keepdim=True)  # 按时间维度求均值
        x = x - mean
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)
        x /= stdev

        yl, yh_ = self.cwt(x.permute(0,2,1))
        yh = yh_[0]
        # plt.figure(figsize=(10, 5))
        # plt.subplot(3, 1, 1)
        # plt.plot(yl.cpu().detach().numpy()[0, 0, :], color='g')
        # plt.subplot(3, 1, 2)
        # plt.plot(yh1.cpu().detach().numpy()[0, 0, :], color='g')
        # plt.subplot(3, 1, 3)
        # plt.plot(yh2.cpu().detach().numpy()[0, 0, :], color='g')
        # plt.show()

        yl_pred = self.yl_linear(yl)
        yh_pred_ = self.yh_linear(yh)
        yh_pred = []
        yh_pred.append(yh_pred_)
        pred = self.icwt((yl_pred, yh_pred))
        pred =self.proj(pred)
        pred = pred.permute(0, 2, 1)
        pred = pred * stdev + mean
        return pred
