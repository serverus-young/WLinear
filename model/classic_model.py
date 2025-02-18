import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.task = config.task
        # 写encoder之前考虑到三个层,第一个EncoderLayer层需要循环n次，代表层数。
        # 第二个AttentionLayer 为了定制Attention，配合各种Att使用
        # 第三个各种att例如self-att，cosatt，qkv先线性化在经过att层
        self.encoder = Encoder(
            EncoderLayer(
                AttentionLayer(
                    FullAttention(mask=True, out_attention=True),
                    d_model=config.d_model, n_heads=config.n_heads
                ),
                d_model=config.d_model, d_ff=config.d_ff, ff_type=config.ff_type, dropout=0.1
            ),
            num_e_layers=config.num_e_layers
        )
        self.enc_embedding = EncEmbedding(d_model=config.d_model, c_in = config.c_in)
        self.decoder = Decoder(
            DecoderLayer(
                AttentionLayer(
                    FullAttention(mask=True, out_attention=False)
                    , d_model=config.d_model, n_heads=config.n_heads
                ),
                AttentionLayer(
                    FullAttention(mask=False, out_attention=False)
                    , d_model=config.d_model, n_heads=config.n_heads
                ),
                d_model=config.d_model, d_ff=config.d_ff, ff_type=config.ff_type
            ),
            num_d_layers=config.num_d_layers
        )
        self.out = nn.Linear(config.d_model,config.c_in)
    def forward(self, x, x_stamp,y,y_stamp):
        if self.task == 'forecast':
            dec_out = self.forecast(x, x_stamp, y, y_stamp)
            return dec_out

    def forecast(self, x, x_stamp, y, y_stamp):
        means = x.mean(1, keepdim=True).detach()
        x_enc = x- means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        enc_in = self.enc_embedding(x)
        enc_out, _ = self.encoder(enc_in,enc_in)
        dec_in = self.enc_embedding(y)
        dec_out = self.decoder(dec_in, enc_out)
        # dec_out.transpose(1, 2)
        dec_out = self.out(dec_out)

        return dec_out

    def classify(self, x):
        pass

    def imputation(self, x):
        pass

    def anomaly_detection(self, x):
        pass


class EncEmbedding(nn.Module):
    def __init__(self, d_model, c_in):
        super(EncEmbedding, self).__init__()
        self.embedding = TokenEmbedding(d_model=d_model, c_in=c_in)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)

    def forward(self, x):
        return self.embedding(x)+self.pos_embedding(x)


class TokenEmbedding(nn.Module):
    def __init__(self, d_model, c_in):
        super(TokenEmbedding, self).__init__()
        self.c_in = c_in
        self.d_model = d_model
        # 卷积核会沿着seq_len这个维度，进行卷积,一个卷积核提取c_in个特征的和。
        # 一共d_model个卷积核
        self.conv1d = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        return x.permute(0, 2, 1)


class PositionalEmbedding(nn.Module):
    # 相对距离越大的输入，其相关性应该越弱
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


class FeedForward(nn.Module):
    # (16,96,516)-(16,96,2048)-(16,96,516)
    def __init__(self, d_model, d_ff, ff_type, dropout=0.1,):
        super(FeedForward, self).__init__()
        assert ff_type in ['linear', 'conv'], 'ff_type should be linear or conv'

        if ff_type == 'linear':
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
        elif ff_type == 'conv':
            self.fc1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
            self.fc2 = nn.Conv1d(d_ff, d_model, kernel_size=1)

        self.dropout = nn.Dropout(dropout)
        self.ff_type = ff_type

    def forward(self, x):
        if self.ff_type == 'conv':
            # conv 需要把后两维交换一下
            # Transpose from [batch_size, seq_len, d_model] to [batch_size, d_model, seq_len] for Conv1d
            x = x.transpose(-1, 1)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(self.fc2(x))
            # Transpose back to [batch_size, seq_len, d_model]
            x = x.transpose(-1, 1)
        else:
            # 先linear-激活函数-dropout-linear-dropout
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(self.fc2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, encoder_layers, num_e_layers):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([encoder_layers for _ in range(num_e_layers)])

    def forward(self, x,x_k):
        attentions = []
        for encoder in self.encoder_layers:
            x, attention = encoder(x,x_k)
            attentions.append(attention)
        return x, attentions


class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, d_ff, ff_type, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention_layer = attention_layer
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = FeedForward(d_model, d_ff, ff_type=ff_type, dropout=dropout)

    def forward(self, x,x_k):
        x, attn = self.attention_layer(x, x_k, x_k)
        # 在浅层模型中，Post-Norm（先残差后归一化）往往效果更好，
        # 而在深层模型中，Pre-Norm（先归一化后残差连接）往往能更好地训练。
        x = self.norm1(x+self.dropout(x))
        x = self.feedforward(x)
        # 经过前馈网络后再Post-Norm最后输出到encoder
        return self.norm2(x+self.dropout(x)), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        # out和attention_weight在att层实现,这一层主要是qkv线性变换，以实现不同注意力机制的封装。
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attention = attention
        self.n_heads = n_heads

    def forward(self, q, k, v):
        B, L, _ = q.shape
        _, S, _ = k.shape
        # (16,96,512)-(16,96,2,256)
        #   # qkv线性重组，学习对于特征。这样的写法前期是 q，k，v的维度都是 d_model/n_heads
        q, k, v = [l(x) for l, x in zip(self.linears, (q, k, v))]
        q = q.view(B, L, self.n_heads, -1)
        k = k.view(B, S, self.n_heads, -1)
        v = v.view(B, S, self.n_heads, -1)
        # out (16,96,2,256)
        out, att = self.attention(q, k, v,)
        # view 操作相当于self-attention的concat操作
        # out (16,96,512)
        out = out.view(B, L, -1)
        # 这里return的是EncodeLayer的输入
        return self.linears[-1](out), att


class FullAttention(nn.Module):
    def __init__(self,out_attention, mask):
        super(FullAttention, self).__init__()
        self.out_attention = out_attention
        self.mask = mask
        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v,attn_mask = None):
        B, l_q, H, E = q.shape
        B, l_k, H, D = k.shape
        B, l_v, H, D = v.shape
        # l×s矩阵对维度 e 进行求和,所以e变成一维的
        att_weight = torch.einsum('blhe,bshe->bhls', q, k)
        # triu创建三角，diagonal=1保证是上三角。B1LL会广播到BHLL
        if self.mask:
            if attn_mask is None:
                with torch.no_grad():
                    attn_mask = torch.triu(torch.ones([B, 1, l_q, l_k], dtype=torch.bool), diagonal=1).to(att_weight.device)
            # 把mask部分用负无穷代替，这样softmax之后就会变成0
            # 注意是masked_fill_不是masked_fill，且在if self.mask之后进行
            att_weight.masked_fill_(attn_mask, -np.inf)
        # Q 和 K 的相似度：表示当前时间步对其他时间步的关注程度。
        # 乘上 V：生成的输出代表了当前时间步的特征，结合了来自其他时间步的信息
        # 在最后一层softmax 随着 E 增大，查询 q 和键 k 的点积值（即 att_weight）会增大。所以引入了缩放因子 1 / sqrt(E)
        att_weight_out = torch.softmax(att_weight / math.sqrt(E), dim=-1)
        att_weight = self.dropout(att_weight_out)
        out = torch.einsum('bhls,bshd-> blhd', att_weight, v)
        if self.out_attention:
            return out.contiguous(), att_weight_out
        else:
            return out.contiguous(), None


class Decoder(torch.nn.Module):
    def __init__(self, DecoderLayer, num_d_layers):
        super(Decoder, self).__init__()
        self.DecoderLayer = nn.ModuleList([DecoderLayer for _ in range(num_d_layers)])

    def forward(self, x, dec_cross):
        for decoder_layer in self.DecoderLayer:
            x = decoder_layer(x, dec_cross)
        return x



class DecoderLayer(nn.Module):
    def __init__(self,self_attention, cross_attention, d_model,d_ff, ff_type, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = FeedForward(d_model, d_ff, ff_type=ff_type)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, ff_type=ff_type)

    def forward(self,dec_in, dec_cross):
        # 原本的transform的decoder是上一次decoder的输出作为下一次输入，知道输出结束符为止。（以机器翻译为例）
        # 所以cross-attention的q代表输出序列上下文整合，k代表输入序列的上下文整合，两个结合得到下一个输出
        # 如果是时间序列任务，滑动窗口的方式decoder一次性输入pred，然后一次性给输出（多步预测）
        # 教师强制（Teacher Forcing）是一种在训练序列到序列模型时常用的策略。·它的主要思想是将真实的目标序列作为当前时间步的输入，以帮助模型更快地学习和收敛。
        dec_out = self.norm1(dec_in + self.dropout(self.self_attention(
            dec_in, dec_in, dec_in)[0]))
        dec_out = self.norm2(dec_out + self.dropout(self.cross_attention(
            dec_in, dec_cross, dec_cross)[0]))
        dec_out = self.norm3(dec_out + self.dropout(self.ff(dec_out)))
        return dec_out


