# -*- coding: utf-8 -*-
"""
DeepTTA Baseline — PyTorch Adaptation
=========================================
Transformer-based drug response prediction (Liang et al. 2023).

Adapted to the unified LCLO feature suite (same conventions as deepcdr.py):
- Cell: gene-expression "barcode" — 2,000 informative genes reshaped to a
  2-D image (40 x 50) and encoded by a 2-layer CNN (DeepTTA's signature
  barcoding front-end).
- Drug: SMILES character sequence encoded by a 2-layer Transformer encoder
  (d_model=64, 4 heads) with mean pooling.
- Fusion: concat -> MLP regression head, MSE loss.

No ID embeddings, no scFoundation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
MAX_LEN = 120


def build_vocab(smiles_list):
    """Character vocabulary from training SMILES + special tokens."""
    chars = set()
    for s in smiles_list:
        chars.update(str(s))
    tokens = [PAD, BOS, EOS] + sorted(chars)
    return {t: i for i, t in enumerate(tokens)}


def encode_smiles(smiles, vocab, max_len=MAX_LEN):
    """Char-level encoding with bos/eos, truncated/padded to max_len."""
    ids = [vocab[BOS]] + [vocab.get(c, 0) for c in str(smiles)[: max_len - 2]] + [vocab[EOS]]
    ids = ids[:max_len]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return np.asarray(ids, dtype=np.int64)


class GeneBarcodeEncoder(nn.Module):
    """2,000 genes -> (1, 40, 50) barcode -> 2-layer CNN -> 256-d."""
    def __init__(self, n_genes=2000, out_dim=256, dropout=0.2):
        super().__init__()
        import math
        self.h = int(math.isqrt(n_genes))
        while n_genes % self.h != 0:
            self.h -= 1
        self.w = n_genes // self.h
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        flat = 64 * (self.h // 4) * (self.w // 4)
        self.proj = nn.Sequential(
            nn.Linear(flat, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        b = x.view(x.size(0), 1, self.h, self.w)
        return self.proj(self.net(b).reshape(x.size(0), -1))


class SmilesTransformer(nn.Module):
    """Char-level Transformer encoder for SMILES -> 256-d."""
    def __init__(self, vocab_size, d_model=64, nhead=4, layers=2,
                 ff=128, out_dim=256, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, ids):
        h = self.emb(ids) + self.pos[:, : ids.size(1)]
        mask = ids == 0                                  # pad mask
        h = self.enc(h, src_key_padding_mask=mask)
        keep = (~mask).unsqueeze(-1).float()
        return self.proj((h * keep).sum(1) / keep.sum(1).clamp(min=1))


class DeepTTA(nn.Module):
    """DeepTTA: gene-barcode CNN + SMILES Transformer -> MLP."""
    def __init__(self, n_genes=2000, vocab_size=64, out_dim=256, dropout=0.2):
        super().__init__()
        self.cell_enc = GeneBarcodeEncoder(n_genes, out_dim, dropout)
        self.drug_enc = SmilesTransformer(vocab_size, out_dim=out_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * out_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(256, 1)

    def forward(self, x_gene, x_smiles):
        h = torch.cat([self.cell_enc(x_gene), self.drug_enc(x_smiles)], dim=-1)
        return self.head(self.fusion(h)).squeeze(-1)


class DeepTTADataset(Dataset):
    def __init__(self, x_gene, smiles_ids, y):
        self.x_gene = torch.FloatTensor(x_gene)
        self.smiles_ids = torch.LongTensor(smiles_ids)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x_gene[i], self.smiles_ids[i], self.y[i]


def train_deeptta(model, tr_ds, te_ds,
                  model_name="deeptta",
                  n_epochs=300, batch_size=1024, lr=1e-3,
                  patience=40, weight_decay=0.01, device="cuda"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=batch_size * 2, shuffle=False,
                           num_workers=0, pin_memory=True)
    total_steps = n_epochs * len(tr_loader)
    warmup = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_rmse, best_state, wait = float("inf"), None, 0
    for epoch in range(n_epochs):
        model.train()
        for xg, xs, yb in tr_loader:
            xg, xs, yb = xg.to(device), xs.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(xg, xs), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xg, xs, yb in te_loader:
                preds.extend(model(xg.to(device), xs.to(device)).cpu().numpy())
                targets.extend(yb.numpy())
        rmse = float(np.sqrt(np.mean((np.array(targets) - np.array(preds)) ** 2)))
        if rmse < best_rmse - 1e-5:
            best_rmse = rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    model.eval()
    preds = []
    with torch.no_grad():
        for xg, xs, yb in te_loader:
            preds.extend(model(xg.to(device), xs.to(device)).cpu().numpy())
    return model, np.array(preds, dtype=np.float32)
