import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert (self.head_dim * nhead == d_model), "d_model must be divisible by nhead"

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        Q = self.linear_q(query).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        K = self.linear_k(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        V = self.linear_v(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.linear_out(attn_output)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.cross_attn = MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        attn_output = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_output))
        attn_output = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout2(attn_output))
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(ff_output))
        return tgt


class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, output_size):
        super(Transformer, self).__init__()
        self.encoder = nn.Linear(d_model, d_model)  # 简化的编码器
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)]
        )
        self.final_layer = nn.Linear(d_model, output_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        memory = self.encoder(memory)
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.final_layer(tgt)
def create_padding_mask(sequence):
    return (sequence != 0).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    return (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1).float().masked_fill(torch.triu(torch.ones(size, size), diagonal=1) == 1, float('-inf'))

def generate_time_series(model, input_sequence, start_value, max_length, end_value):
    generated = [start_value]
    model.eval()
    enc_padding_mask = create_padding_mask(input_sequence)

    with torch.no_grad():
        enc_output = model.encoder(input_sequence)

    for _ in range(max_length):
        dec_input = torch.tensor(generated).unsqueeze(0)
        look_ahead_mask = create_look_ahead_mask(dec_input.size(1))
        dec_padding_mask = create_padding_mask(dec_input)

        with torch.no_grad():
            output = model(dec_input, enc_output, look_ahead_mask, dec_padding_mask)
            next_value = torch.argmax(output[:, -1, :], dim=-1).item()
            generated.append(next_value)

            if next_value == end_value:
                break

    return generated
# 示例参数
d_model = 512
nhead = 4
num_layers = 4
dim_feedforward = 256
output_size = 10  # 假设目标序列是10维的

# 创建模型
model = Transformer(d_model, nhead, num_layers, dim_feedforward, output_size)

# 假设输入序列和参数
input_sequence = torch.rand(1, 96, d_model)  # (batch_size, sequence_length, d_model)
start_value = 0  # 自定义的起始值
end_value = 9    # 自定义的结束值
max_length = 20  # 最大生成长度

# 生成时间序列
generated_series = generate_time_series(model, input_sequence, start_value, max_length, end_value)

print("Generated Time Series:", generated_series)
