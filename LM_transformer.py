import math

from torch import nn
import torch
from simple_transformer import SimpleTransformer


class LM_factor(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, dim_feedforward=512, dropout=0.1):
        super(LM_factor, self).__init__()
        self.d_model = d_model
        self.Transformer = SimpleTransformer(vocab_size, d_model, nhead, dim_feedforward, dropout)
        self.res=nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.GELU(),
        )
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self,x):
        x1 = self.Transformer.forward(x)
        res = self.res(x1)
        x1 = res + x1
        x1 = self.layer_norm(x1)
        return x1

    def step_generate(self, x):
        x1 = self.Transformer.forward(x)
        res = self.res(x1)
        x1 = res + x1
        x1 = self.layer_norm(x1)
        return x1
class LM_last(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, dim_feedforward=512, dropout=0.1):
        super(LM_last, self).__init__()
        self.d_model = d_model
        self.Transformer = SimpleTransformer(vocab_size, d_model, nhead, dim_feedforward, dropout)
        self.res=nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.GELU(),
        )
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self,x):
        x1 = self.Transformer.forward(x)
        res = self.res(x1)
        x1 = res + x1
        return x1

    def step_generate(self, x):
        x1 = self.Transformer.forward(x)
        res = self.res(x1)
        x1 = res + x1
        return x1
class LM(nn.Module):
    def __init__(self, vocab_size, cengshu=8, d_model=128, nhead=8, dim_feedforward=512, dropout=0.1):
        super(LM, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=0.00)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 10000, d_model) * 0.1)  # 最大长度5000
        self.cengshu = cengshu
        self.LM_lastone = LM_last(vocab_size, d_model, nhead, dim_feedforward, dropout)
        self.LM_total = nn.ModuleList()
        for i in range(cengshu):
            self.LM_total.append(LM_factor(vocab_size, d_model, nhead, dim_feedforward, dropout))
        self.output_layer = nn.Linear(d_model, vocab_size)
    def forward(self,x):
        seq_len = x.size(1)
        embed = self.embedding(x) * math.sqrt(self.d_model)  # 缩放
        embed = embed + self.pos_encoder[:, :seq_len, :]
        if self.cengshu == 1:
            embed = self.LM_lastone(embed)
        else:
            for i in range(len(self.LM_total)):
                embedi = self.LM_total[i](embed)
                embed = embedi
            embed = self.LM_lastone(embed)
        embed = self.output_layer(embed)
        embed = self.dropout(embed)
        return embed
    def step_generate(self, x):
        seq_len = x.size(1)
        embed = self.embedding(x) * math.sqrt(self.d_model)  # 缩放
        embed = embed + self.pos_encoder[:, :seq_len, :]
        if self.cengshu == 1:
            embed = self.LM_lastone(embed)
        else:
            for i in range(len(self.LM_total)):
                embedi = self.LM_total[i].step_generate(embed)
                embed = embedi
            embed = self.LM_lastone(embed)
        embed = self.output_layer(embed)
        return embed
