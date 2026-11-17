# -*- coding: utf-8 -*-
"""
MOLI Baseline — PyTorch Adaptation
=========================================
Multi-Omics Late Integration (Sharifi-Noghabi et al., Bioinformatics 2019).

Adapted to the unified LCLO feature suite (same conventions as deepcdr.py):
- Cell: three late-integration branches — gene expression (2,000),
  binary driver mutations (448), GISTIC CNA — each projected to 256-d
  with BatchNorm + dropout, then concatenated (MOLI's signature
  late-integration SNN).
- Drug: Morgan fingerprint (2,048) branch projected to 256-d
  (MOLI used chemical descriptors; fingerprints keep the suite identical).
- Loss: MOLI "canonball" loss = beta * MSE + (1 - beta) * (1 - pearson).

No ID embeddings, no scFoundation (MOLI pre-dates both).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class OmicsBranch(nn.Module):
    """Single omics branch: Linear -> BN -> ReLU -> Dropout."""
    def __init__(self, in_dim, out_dim=256, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MOLI(nn.Module):
    """MOLI: late integration of expression + mutation + CNA + drug FP."""
    def __init__(self, n_genes=2000, n_mut=448, n_cna=0, fp_bits=2048,
                 branch_dim=256, dropout=0.5, head_dropout=0.3):
        super().__init__()
        self.exp_branch = OmicsBranch(n_genes, branch_dim, dropout)
        self.mut_branch = OmicsBranch(n_mut, branch_dim, dropout) if n_mut > 0 else None
        self.cna_branch = OmicsBranch(n_cna, branch_dim, dropout) if n_cna > 0 else None
        self.drug_branch = OmicsBranch(fp_bits, branch_dim, dropout)

        n_branches = 2 + (1 if self.mut_branch else 0) + (1 if self.cna_branch else 0)
        fuse_in = n_branches * branch_dim
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
        )
        self.head = nn.Linear(256, 1)

    def forward(self, x_gene, x_mut, x_cna, x_fp):
        parts = [self.exp_branch(x_gene), self.drug_branch(x_fp)]
        if self.mut_branch is not None:
            parts.append(self.mut_branch(x_mut))
        if self.cna_branch is not None:
            parts.append(self.cna_branch(x_cna))
        h = self.fusion(torch.cat(parts, dim=-1))
        return self.head(h).squeeze(-1)


class MOLIDataset(Dataset):
    """x_mut / x_cna may be None (organoid transfer: views unavailable)."""
    def __init__(self, x_gene, x_mut, x_cna, x_fp, y):
        n = len(y)
        self.x_gene = torch.FloatTensor(x_gene)
        self.x_mut = torch.zeros(n, 1) if x_mut is None else torch.FloatTensor(x_mut)
        self.x_cna = torch.zeros(n, 1) if x_cna is None else torch.FloatTensor(x_cna)
        self.x_fp = torch.FloatTensor(x_fp)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x_gene[i], self.x_mut[i], self.x_cna[i], self.x_fp[i], self.y[i]


def _canonball_loss(pred, target, beta=0.5, eps=1e-8):
    """MOLI loss: beta * MSE + (1 - beta) * (1 - Pearson r)."""
    mse = F.mse_loss(pred, target)
    vp = pred - pred.mean()
    vt = target - target.mean()
    corr = (vp * vt).sum() / (vp.norm() * vt.norm() + eps)
    return beta * mse + (1.0 - beta) * (1.0 - corr)


def train_moli(model, tr_ds, te_ds,
               model_name="moli",
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
        for xg, xm, xc, xf, yb in tr_loader:
            xg, xm, xc, xf, yb = (t.to(device) for t in (xg, xm, xc, xf, yb))
            optimizer.zero_grad()
            out = model(xg, xm, xc, xf)
            loss = _canonball_loss(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xg, xm, xc, xf, yb in te_loader:
                xg, xm, xc, xf = (t.to(device) for t in (xg, xm, xc, xf))
                preds.extend(model(xg, xm, xc, xf).cpu().numpy())
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
        for xg, xm, xc, xf, yb in te_loader:
            xg, xm, xc, xf = (t.to(device) for t in (xg, xm, xc, xf))
            preds.extend(model(xg, xm, xc, xf).cpu().numpy())
    return model, np.array(preds, dtype=np.float32)
