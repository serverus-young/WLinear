import torch
import torch.nn as nn
from pytorch_wavelets import DWT1D, IDWT1D  # 一维离散小波变换
from matplotlib import pyplot as plt
from torch.nn import MSELoss

from utils.metric  import metric
import numpy


class WFT(nn.Module):
    def __init__(self, args):
        super(WFT, self).__init__()
        self.configs = args
        # 配置 DWT1D 和 IDWT1D
        self.cwt = DWT1D(wave='db3', mode='zero' )  # 一维离散小波正变换
        self.icwt = IDWT1D(wave='db3', mode='zero')  # 一维离散小波逆变换
        self.yh_linear = nn.Linear(args.seq_len//2 + 2, args.pred_len//2 + 2)
        self.yl_linear = nn.Linear(args.seq_len//2 + 2, args.pred_len//2 + 2)
    def forward(self, x, x_stamp, y, y_stamp):
        """
        前向传播
        """
        # x 的维度为 (b, l, n)
        b, l, n = x.size()

        # 数据标准化
        mean = x.mean(dim=1, keepdim=True)  # 按时间维度求均值
        x = x - mean
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)
        x /= stdev

        mean = y.mean(dim=1, keepdim=True)  # 按时间维度求均值
        y = y - mean
        stdev = torch.sqrt(torch.var(y, dim=1, keepdim=True) + 1e-5)
        y /= stdev
        # 调整输入形状为 (b, n, l) 以适配 DWT1D
        x = x.permute(0, 2, 1)  # 将形状从 (b, l, n) 变为 (b, n, l)
        y = y.permute(0, 2, 1)

        y = y[:, :, -self.configs.pred_len:]
        y_yl,y_yh = self.cwt(y)
        y_yh1 = y_yh[0]
        # 小波正变换
        Yl, Yh = self.cwt(x)  # 返回低频分量 (Yl) 和高频分量 (Yh)
        yh = Yh[0]        # 展平高频分量并输入线性层

        yh_pred = self.yh_linear(yh)
        Yl_pred = self.yl_linear(Yl)


        # plt.figure(figsize=(10, 5))
        # plt.plot(Yl.cpu().detach().numpy()[0, 0, :], color='g')
        # plt.plot(yh.cpu().detach().numpy()[0, 0, :], color='r')
        #
        # plt.figure(figsize=(10, 5))
        # plt.subplot(3, 1, 1)
        # plt.plot(Yl_pred.cpu().detach().numpy()[0, 0, :], color='g')
        # plt.plot(y_yl.cpu().detach().numpy()[0, 0, :], color='r')
        # plt.subplot(3, 1, 2)
        # plt.plot(yh_pred.cpu().detach().numpy()[0, 0, :], color='g')
        # plt.plot(y_yh1.cpu().detach().numpy()[0, 0, :], color='r')
        # plt.show()
        # 逆变换
        Yh_pred = []
        Yh_pred.append(yh_pred)
        pred = self.icwt((Yl_pred, Yh_pred))  # 逆变换后的形状为 (b, n, l)
        # plt.subplot(3, 1, 3)
        # plt.plot(y.cpu().detach().numpy()[0, 0, :], color='g')
        # plt.plot(pred.cpu().detach().numpy()[0, 0, :], color='r')
        # plt.show()
        # 恢复形状为 (b, l, n)
        pred = pred.permute(0, 2, 1)
        # y = y.permute(0, 2, 1)
        # yh_MSE, MAE, MAPE, MSPE, yh_RMSE = metric(yh_pred.detach().cpu().numpy(), y_yh1.detach().cpu().numpy())
        # print('yhRMSE: ', yh_RMSE)
        # yl_MSE, MAE, MAPE, MSPE, yl_RMSE = metric(Yl_pred.detach().cpu().numpy(), y_yl.detach().cpu().numpy())
        # print('ylRMSE: ', yl_RMSE)
        # 还原标准化
        pred = pred * stdev + mean
        return pred