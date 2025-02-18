import torch.nn as nn
import torch
import math
class dataembedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.kernel_size = args.kernel_size
        self.stride = args.stride
        self.padding = args.padding
        self.dropout = args.dropout
        self.token_embedding = token_embedding(self.input_size, self.kernel_size, self.stride, self.padding)
        self.dropout = nn.Dropout(self.dropout)
    def forward(self, x,x_emb):
        x = self.token_embedding(x)
        x_stamp = self.token_embedding(x_emb)
        return self.dropout(x),self.dropout(x_stamp)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class token_embedding(nn.Module):
    def __init__(self,input_size,output_size,kernel_size,stride,padding):
        super().__init__()
        # kernel_size =3 + padding = 1 保证了卷积完序列长度还是96，从6维变成512维度
        self.conv = nn.Conv1d(input_size,output_size,kernel_size=kernel_size,stride=stride,padding=padding)
    # 这一段代码的作用是对模型中的所有 nn.Conv1d 一维卷积层的权重进行 Kaiming 初始化。这样做的目的是为了：
    #
    #     确保更好的训练表现： Kaiming 初始化有助于保持梯度的方差稳定，防止梯度消失或爆炸，从而加速网络训练。
    #     提高网络的收敛速度： 正确初始化权重有助于更快地收敛到最优解。
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self,x):
        x = self.conv(x.permute(0,2,1)).permute(0,2,1)
        return x
class timestamp_embedding(nn.Module):
    def __init__(self,input_size,output_size,kernel_size,stride,padding):
        super().__init__()
        self.hour = 24
        self.weekday = 7
        self.day = 31
        self.month = 12
        self.embed = nn.Embedding(input_size,output_size)
    def forward(self,x):
        x =x .long()
        x_emc_hour = self.embed(x[:,:,3])
        x_emc_weekday = self.embed(x[:,:,2])
        x_emc_day = self.embed(x[:,:,1])
        x_emc_month = self.embed(x[:,:,0])
        return x_emc_day + x_emc_hour + x_emc_weekday + x_emc_month

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, stride, padding, d_model, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.padding = nn.ReplicationPad1d((0, padding))
        self.linear= nn.Linear(patch_size, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.padding(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # （112,12,512）
        x = x.reshape(-1,x.shape[2],x.shape[3])

        x = self.linear(x) + self.position_embedding(x)
        return self.dropout(x)


import torch
import torch.nn as nn


class Patch3Embedding(nn.Module):
    def __init__(self, patch_size, stride, d_model, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.conv = nn.Conv1d(in_channels=7, out_channels=d_model, kernel_size=patch_size, stride=stride)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x 的形状是 (batch_size, seq_len, num_features)，首先进行维度转换
        x = x.permute(0, 2, 1)  # (batch_size, num_features, seq_len)

        # 使用卷积提取 patch
        x = self.conv(x)  # (batch_size, d_model, num_patches)

        # 添加位置编码
        x = x.permute(0, 2, 1)  # (batch_size, num_patches, d_model)
        x = x + self.position_embedding(x)

        # Dropout
        return self.dropout(x)


class Patch2Embedding(nn.Module):
    def __init__(self, patch_size, stride, padding, d_model, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.padding = nn.ReplicationPad1d((0, padding))
        self.linear= nn.Linear(patch_size*7, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.padding(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # （16,12,512）
        x = x.reshape(x.shape[0],x.shape[2],-1)

        x = self.linear(x) + self.position_embedding(x)
        return self.dropout(x)