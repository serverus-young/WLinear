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
import torch
import torch.nn as nn
from pytorch_wavelets import DWT1D, IDWT1D  # 一维离散小波变换
from matplotlib import pyplot as plt
from torch.nn import MSELoss

from utils.metric  import metric
import numpy
class WTcore(nn.Module):
    def __init__(self,level, input_size, output_size):
        super(WTcore, self).__init__()
        # 配置 DWT1D 和 IDWT1D
        self.level = level
        self.cwt = DWT1D(J=level, wave='db3', mode='zero')  # 一维离散小波正变换
        self.icwt = IDWT1D(wave='db3', mode='zero')  # 一维离散小波逆变换
        if level == 3:
            self.yl_linear = nn.Linear(input_size // 8 + 4, output_size // 8 + 4)
            self.yh1_linear = nn.Linear(input_size // 2 + 2, output_size // 2 + 2)
            self.yh2_linear = nn.Linear(input_size // 4 + 3, output_size // 4 + 3)
            self.yh3_linear = nn.Linear(input_size // 8 + 4, output_size // 8 + 4)
        elif level == 2:
            self.yl_linear = nn.Linear(input_size // 4 + 3, output_size // 4 + 3)
            self.yh1_linear = nn.Linear(input_size // 2 + 2, output_size // 2 + 2)
            self.yh2_linear = nn.Linear(input_size // 4 + 3, output_size // 4 + 3)
        elif level == 1:
            self.yl_linear = nn.Linear(input_size // 2 + 2, output_size // 2 + 2)
            self.yh_linear = nn.Linear(input_size // 2 + 2, output_size // 2 + 2)
    def forward(self, x):
        """
        前向传播
        """
        # x 的维度为 (b, l, n)
        yl, yh_ = self.cwt(x.permute(0,2,1))
        if self.level == 1:
            yh = yh_[0]
            yl_pred = self.yl_linear(yl)
            yh_pred_ = self.yh_linear(yh)
            yh_pred = []
            yh_pred.append(yh_pred_)
        elif self.level == 2 :
            yh1 = yh_[0]
            yh2 = yh_[1]
            yl_pred = self.yl_linear(yl)
            yh1_pred = self.yh1_linear(yh1)
            yh2_pred = self.yh2_linear(yh2)
            yh_pred = []
            yh_pred.append(yh1_pred)
            yh_pred.append(yh2_pred)
        elif self.level == 3:
            yh1 = yh_[0]
            yh2 = yh_[1]
            yh3 = yh_[2]
            yl_pred = self.yl_linear(yl)
            yh1_pred = self.yh1_linear(yh1)
            yh2_pred = self.yh2_linear(yh2)
            yh3_pred = self.yh3_linear(yh3)
            yh_pred = []
            yh_pred.append(yh1_pred)
            yh_pred.append(yh2_pred)
            yh_pred.append(yh3_pred)
        pred = self.icwt((yl_pred, yh_pred))
        return pred

class WFT(nn.Module):
    def __init__(self, args):
        super(WFT, self).__init__()
        self.configs = args
        self.level = args.level
        self.individual = args.individual
        # 配置 DWT1D 和 IDWT1D
        self.WTcore = WTcore(args.level, args.seq_len, args.d_model)
        if self.individual:
            self.proj = nn.ModuleList()
            for i in range(args.e_in):
                self.proj.append(nn.Linear(args.d_model, args.pred_len))
        else:
            self.proj = nn.Linear(args.d_model, args.pred_len)
        self.Tanh = nn.Tanh()
    def forward(self, x, x_stamp, y, y_stamp):
        mean = x.mean(dim=1, keepdim=True)  # 按时间维度求均值
        x = x - mean
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)
        x /= stdev

        pred = self.WTcore(x)
        pred = self.Tanh(pred)
        if self.individual:
            pred_ = torch.zeros([pred.size(0), pred.size(1),self.configs.pred_len], dtype=pred.dtype, device=pred.device)
            for i in range(pred.size(1)):
                pred_[:, i, :] = self.proj[i](pred[:, i, :])
        else:
            pred_ =self.proj(pred)

        pred_ = pred_.permute(0, 2, 1)
        pred_ = pred_ * stdev + mean

        # plt.figure(figsize=(10, 5))
        # plt.subplot(4, 1, 1)
        # plt.plot(x.permute(0, 2, 1).cpu().detach().numpy()[0, 0, :], color='g')
        # plt.subplot(4, 1, 2)
        # plt.plot(pred.cpu().detach().numpy()[0, 0, :], color='g')
        # plt.subplot(4, 1, 3)
        # plt.plot(pred_.permute(0, 2, 1).cpu().detach().numpy()[0, 0, :], color='g')
        # plt.plot(y.permute(0, 2, 1).cpu().detach().numpy()[0, 0, -self.configs.pred_len:], color='r')
        # plt.show()
        return pred_
