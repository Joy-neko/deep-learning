import math
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_head == 0

        self.n_head = n_head
        self.d_model = d_model
        self.d_tensor = d_model // n_head  # 每个头的维度

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_concat = nn.Linear(d_model, d_model)  # 对不同头的信息进行线性融合

    def split(self, tensor):  # 将 d_model 拆成 n_head 个独立的头
        batch_size, seq_len, _ = tensor.size()
        tensor = tensor.view(
            batch_size,
            seq_len,
            self.n_head,
            self.d_tensor
        )
        return tensor.transpose(1, 2)

    def concat(self, tensor):  # 将多个头物理拼接回来，之后再由 w_concat 做线性融合
        batch_size, _, seq_len, _ = tensor.size()
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(
            batch_size,
            seq_len,
            self.d_model
        )

    def forward(self, q, k, v, mask=None):  # 参数分开是为了同时支持 self-attention 和 cross-attention
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = self.split(q)
        k = self.split(k)
        v = self.split(v)

        # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
        score = torch.matmul(q, k.transpose(-2, -1))
        score = score / math.sqrt(self.d_tensor)

        if mask is not None:
            score = score.masked_fill(~mask, -1e9)

        score = F.softmax(score, dim=-1)
        out = torch.matmul(score, v)

        out = self.concat(out)
        out = self.w_concat(out)
        return out

