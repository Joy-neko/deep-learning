import torch
from torch import nn

from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_pad_idx,
        trg_pad_idx,
        enc_voc_size,
        dec_voc_size,
        d_model,
        n_head,
        max_len,
        ffn_hidden,
        n_layers,
        drop_prob,
        device
    ):
        super(Transformer, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder = Encoder(
            enc_voc_size=enc_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            n_layers=n_layers,
            drop_prob=drop_prob,
            pad_idx=src_pad_idx,
            device=device
        )

        self.decoder = Decoder(
            dec_voc_size=dec_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            n_layers=n_layers,
            drop_prob=drop_prob,
            pad_idx=trg_pad_idx,
            device=device
        )

    def make_src_mask(self, src):
        # [B, Lsrc] -> [B, 1, 1, Lsrc]
        src_mask = (src != self.src_pad_idx)
        return src_mask.unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        # padding mask: [B, Ltgt] -> [B, 1, 1, Ltgt]
        trg_pad_mask = (trg != self.trg_pad_idx)
        trg_pad_mask = trg_pad_mask.unsqueeze(1).unsqueeze(2)

        trg_len = trg.size(1)

        # causal mask: [Ltgt, Ltgt] -> [1, 1, Ltgt, Ltgt]
        trg_sub_mask = torch.tril(
            torch.ones(
                trg_len,
                trg_len,
                device=trg.device,
                dtype=torch.bool
            )
        )
        trg_sub_mask = trg_sub_mask.unsqueeze(0).unsqueeze(1)

        # 广播后得到 [B, 1, Ltgt, Ltgt]
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(
            trg,
            enc_src,
            trg_mask,
            src_mask
        )
        return output

