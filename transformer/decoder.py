from torch import nn

from attention import MultiHeadAttention
from embedding import TransformerEmbedding
from encoder import LayerNorm, PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(
            d_model,
            ffn_hidden,
            drop_prob
        )
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, enc_src, trg_mask, src_mask):
        residual = x
        x = self.self_attention(
            q=x,
            k=x,
            v=x,
            mask=trg_mask
        )
        x = self.dropout1(x)
        x = self.norm1(x + residual)

        residual = x
        x = self.cross_attention(
            q=x,
            k=enc_src,
            v=enc_src,
            mask=src_mask
        )
        x = self.dropout2(x)
        x = self.norm2(x + residual)

        residual = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + residual)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        dec_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layers,
        drop_prob,
        pad_idx,
        device
    ):
        super(Decoder, self).__init__()

        self.emb = TransformerEmbedding(
            dec_voc_size,
            d_model,
            max_len,
            drop_prob,
            pad_idx,
            device
        )

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model,
                ffn_hidden,
                n_head,
                drop_prob
            )
            for _ in range(n_layers)
        ])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        x = self.emb(trg)

        for layer in self.layers:
            x = layer(
                x,
                enc_src,
                trg_mask,
                src_mask
            )

        output = self.linear(x)
        return output

