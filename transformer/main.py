import torch

from transformer import Transformer



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Transformer(
    src_pad_idx=1,
    trg_pad_idx=1,
    enc_voc_size=100,
    dec_voc_size=120,
    d_model=32,
    n_head=4,
    max_len=100,
    ffn_hidden=128,
    n_layers=2,
    drop_prob=0.1,
    device=device
).to(device)

src = torch.tensor([
    [2, 5, 7, 9, 1, 1],
    [4, 8, 3, 6, 7, 1]
], device=device)

trg = torch.tensor([
    [2, 10, 20, 30, 1],
    [2, 11, 21, 31, 41]
], device=device)

output = model(src, trg)
print('output shape:', output.shape)  # 期望: torch.Size([2, 5, 120])
