import math
import torch
from torch import nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model, pad_idx):
        super(TokenEmbedding, self).__init__(
            vocab_size,
            d_model,
            padding_idx=pad_idx
        )


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEmbedding, self).__init__()

        encoding = torch.zeros(max_len, d_model, device=device)
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        # 位置编码不是可训练参数，但属于模型状态；register_buffer 会让它随 model.to(device) 一起移动
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        _, seq_len = x.size()
        return self.encoding[:seq_len, :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, pad_idx, device):
        super(TransformerEmbedding, self).__init__()

        self.d_model = d_model
        self.tok_emb = TokenEmbedding(vocab_size, d_model, pad_idx)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 原论文中 token embedding 会乘 sqrt(d_model)，再与位置编码相加
        tok_emb = self.tok_emb(x) * math.sqrt(self.d_model)
        pos_emb = self.pos_emb(x)
        return self.dropout(tok_emb + pos_emb)

