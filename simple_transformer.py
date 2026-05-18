import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
class SimpleTransformer(nn.Module):
    """单层 Transformer 语言模型（因果自回归）"""
    def __init__(self, vocab_size, d_model=128, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        #self.embedding = nn.Embedding(vocab_size, d_model)
        # 可学习位置编码
        #self.pos_encoder = nn.Parameter(torch.randn(1, 10000, d_model) * 0.1)  # 最大长度5000
        # 单层 Transformer Encoder（需手动应用因果掩码）
        self.encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # 输入形状 (batch, seq, d_model)
        )
        self.decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        #self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: (batch, seq_len)  token ids
        返回: (batch, seq_len, vocab_size) logits
        """
        seq_len = x.size(1)
        #embed = self.embedding(x) * math.sqrt(self.d_model)  # 缩放
        #embed = embed + self.pos_encoder[:, :seq_len, :]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        memory = self.encoder_layer(x, src_mask=mask)
        output = self.decoder_layer(x, memory)
        #output = self.output_layer(output)
        return output
