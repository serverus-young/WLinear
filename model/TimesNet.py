import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

class Inception_Block(nn.Module):
    # 更复杂的模块，包含多个卷积层，因此需要在初始化时对权重进行设置以保证更好的训练收敛性。
    def __init__(self,in_channels,out_channels,num_kernels,init_weights=True):
        super(Inception_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernel = []
        for i in range(num_kernels):
            kernel.append(nn.Conv2d(in_channels,out_channels,kernel_size=2*i+1,stride=1,padding=i))
        self.kernel = nn.ModuleList(kernel)
        if init_weights:
            self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

    def forward(self,x):
        #  padding=(1, 1) 表示在输入张量的每一侧（上、下、左、右） 有五层conv2d，卷积核1-5，padding1-5，
        # 不同大小的卷积核可以有效地捕捉到不同尺度的特征。
        # 例如，小的卷积核（如 1×11×1 或 3×33×3）可以提取细节特征，
        # 而较大的卷积核（如 5×55×5 或 7×77×7）则可以捕捉更大的上下文信息。
        # 每一层卷积得到均为（16,16,64,3）进过五层堆叠变成（16,16,64,3,5），再在最后一层取均值
        res = []
        for i in range(self.num_kernels):
            res.append(self.kernel[i](x))
        res = torch.stack(res,dim =-1).mean(-1)
        return res
class TimeBlock(nn.Module):
    def __init__(self,config):
        super(TimeBlock, self).__init__()
        self.config = config
        self.inception_blocks = nn.Sequential(
            Inception_Block(config.d_model,config.d_ff,config.num_kernels),
            nn.GELU(),
            Inception_Block(config.d_ff,config.d_model,config.num_kernels)
        )
    def forward(self,x):
        # FFT for periods
        B,L,C = x.size()
        periods,per_weight = FFT_for_periods(x, self.config.k)
        res = []
        # resahpe (bs -1 period d_model )
        for i in range(self.config.k):
            period = periods[i]
            # padding
            if (self.config.seq_len + self.config.pred_len) % period != 0:
                length = (
                                 ((self.config.seq_len + self.config.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.config.seq_len + self.config.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.config.seq_len + self.config.pred_len)
                out = x
            out = out.reshape(B, length // period, period,
                              C).permute(0, 3, 1, 2).contiguous()
            #     转换成（16，16,96,2）输入给Inception_Block 得到 （16,num_kernel,96,2)
            # Inception_Block
            out = self.inception_blocks(out)
            out = out.permute(0,2,3,1).reshape(B,-1,C)
            res.append(out[:,:(self.config.seq_len+self.config.pred_len),:])
        res = torch.stack(res,dim = -1)
        per_weight = F.softmax(per_weight,dim =1)
        per_weight = per_weight.unsqueeze(1).unsqueeze(1).repeat(1,self.config.seq_len+self.config.pred_len,C,1)
        res = torch.sum(per_weight*res,-1)
        res = res + x
        return res,periods

def FFT_for_periods(x,k):
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    # abs 是对复数取模 192 得到 96个频率分量
    frequency_list = abs(xf).mean(0).mean(-1)
    # 这里为什么要把第一个频率分量设置为0？？：直流分量不参与计算
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    # top_list可以理解为计算自相关系数 x平移的长度
    period = x.shape[1] // top_list
    # 返回最大频率的复数 实部表示信号在对应频率上的余弦成分 虚部表示信号在对应频率上的正弦成分
    return period, abs(xf).mean(-1)[:, top_list]


class TimesNet(nn.Module):
    def __init__(self,config):
        super(TimesNet, self).__init__()
        self.config = config
        self.embedding = DataEmbedding(c_in=config.c_in, d_model=config.d_model)
        self.predict_linear = nn.Linear(config.seq_len, config.seq_len+config.pred_len)
        self.TimeBlocks = nn.ModuleList([TimeBlock(config) for _ in range(config.num_e_layers)])
        self.out_linear = nn.Linear(config.d_model, config.c_in,bias=True)
        self.lay_norm = nn.LayerNorm(config.d_model)
    def forward(self, x,x_stamp,y,y_stamp):
        time1 = time.time()
        # stationary
        mean = x.mean(dim=1, keepdim=True).detach()
        x_enc = x - mean
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False)+1e-5)
        x_enc /= stdev
        # embedding ->(bs,seq+pred,d_model)
        x_enc = self.embedding(x_enc,x_stamp)
        x_enc = self.predict_linear(x_enc.permute(0,2,1)).permute(0,2,1)
        # TimeBlock
        period = []
        for i in range(self.config.num_e_layers):
            # block_time = time.time()
            x_enc,periods = self.TimeBlocks[i](x_enc)
            x_enc =self.lay_norm(x_enc)
            # time_cost = time.time() - block_time
        # print(periods)
        out = self.out_linear(x_enc)
        # de stationary
        dec_out = out * stdev.repeat(1, self.config.pred_len+self.config.seq_len,1)
        dec_out = dec_out + mean.repeat(1, self.config.pred_len+self.config.seq_len,1)
        # dec_out = dec_out[:,self.config.pred_len,:]
        time1 = time.time() - time1
        # print(time1)
        return dec_out