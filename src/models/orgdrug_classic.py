# -*- coding: utf-8 -*-
"""
OrgDrug-Classic — GDRNet v1 backbone + OrgDrug feature blocks (surgical upgrade)
=================================================================================
Keeps v1's proven recipe EXACTLY (BN-input MLP encoders, ReLU, concat, BN+Linear
input_proj, DCNv2 x3, residual MLP, same head) and surgically adds:
  * mutation set encoder (shared gene embeddings, attention-pooled, 64-d)
  * pathway score encoder (51 -> 64)
  * drug target set encoder (64-d, same shared embedding table)
  * optional: extreme weighting / listnet / aux-cls loss (off by default)
Missing modalities: unavailable views contribute zeros + a flag dim (v1-style
simple, no attention masking).

This is the direct "GDRNet upgraded" variant: identical method, upgraded inputs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEnc(nn.Module):
    """v1-exact MLP encoder: BN -> Linear -> ReLU -> Dropout -> Linear."""

    def __init__(self, in_dim, hidden, out_dim, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        nn.init.kaiming_normal_(self.net[1].weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.net[4].weight, nonlinearity="linear")

    def forward(self, x):
        return self.net(x)


class SetEnc(nn.Module):
    """Attention-pooled set encoder over a shared gene embedding table (64-d out)."""

    def __init__(self, vocab_size, d_emb=32, d_out=64, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, d_emb, padding_idx=0)
        nn.init.normal_(self.emb.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[0].zero_()
        self.proj = nn.Linear(d_emb, d_out)
        self.query = nn.Parameter(torch.randn(1, 1, d_out) * 0.02)
        self.attn = nn.MultiheadAttention(d_out, n_heads, batch_first=True)

    def forward(self, idx, mask):
        e = self.proj(self.emb(idx))
        key_pad = ~mask
        empty = ~mask.any(dim=1, keepdim=True)
        key_pad = key_pad & ~empty
        q = self.query.expand(idx.size(0), -1, -1)
        a, _ = self.attn(q, e, e, key_padding_mask=key_pad)
        return a.squeeze(1) * mask.any(dim=1, keepdim=True).float()


class ResBlock(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.bn = nn.BatchNorm1d(d_out)
        self.drop = nn.Dropout(dropout)
        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")

    def forward(self, x):
        return self.drop(F.relu(self.bn(self.fc(x)))) + self.shortcut(x)


class CrossLayer(nn.Module):
    def __init__(self, dim, rank=64):
        super().__init__()
        self.U = nn.Linear(dim, rank, bias=False)
        self.V = nn.Linear(rank, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, x0, xl):
        return x0 * self.V(F.relu(self.U(xl))) + self.bias + xl


class OrgDrugClassic(nn.Module):
    """v1 layout with extra blocks:
    cell = [gene 256, scf 256, mut 64, path 64, cell_id 64, mut_flag 1, path_flag 1]
    drug = [fp 256, tgt 64, drug_id 64, tgt_flag 1]
    concat -> BN -> Linear -> ReLU (input_proj, 512) -> DCNv2 x3 + ResMLP -> head
    """

    def __init__(self, n_genes=2000, scf_dim=3072, fp_dim=2239, path_dim=51,
                 n_shared_genes=519, n_cells=965, n_drugs=230,
                 d_hidden=256, id_emb_dim=64, n_cross=3, cross_rank=64,
                 dropout=0.15):
        super().__init__()
        self.gene_enc = MLPEnc(n_genes, 512, d_hidden, dropout)
        self.scf_enc = MLPEnc(scf_dim, 512, d_hidden, dropout)
        self.path_enc = MLPEnc(path_dim, 128, 64, dropout)
        self.mut_enc = SetEnc(n_shared_genes)
        self.fp_enc = MLPEnc(fp_dim, 512, d_hidden, dropout)
        self.tgt_enc = SetEnc(n_shared_genes)
        self.cell_emb = nn.Embedding(n_cells + 1, id_emb_dim, padding_idx=0)
        self.drug_emb = nn.Embedding(n_drugs + 1, id_emb_dim, padding_idx=0)
        nn.init.normal_(self.cell_emb.weight, std=0.02)
        nn.init.normal_(self.drug_emb.weight, std=0.02)

        cell_dim = d_hidden * 2 + 64 * 3 + 2      # 770
        drug_dim = d_hidden + 64 * 2 + 1          # 385
        self.input_proj = nn.Sequential(
            nn.BatchNorm1d(cell_dim + drug_dim),
            nn.Linear(cell_dim + drug_dim, 512),
            nn.ReLU(inplace=True),
        )
        self.cross_layers = nn.ModuleList(
            [CrossLayer(512, rank=cross_rank) for _ in range(n_cross)])
        self.deep_layers = nn.ModuleList(
            [ResBlock(a, b, dropout) for a, b in [(512, 256), (256, 128), (128, 64)]])
        self.head = nn.Sequential(
            nn.Linear(512 + 64, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x_gene, scf, x_fp, path, mut_idx, mut_mask,
                tgt_idx, tgt_mask, cell_idx, drug_idx, avail):
        h_gene = self.gene_enc(x_gene)
        h_scf = self.scf_enc(scf)
        h_path = self.path_enc(path) * avail[:, 3:4]
        h_mut = self.mut_enc(mut_idx, mut_mask) * avail[:, 2:3]
        h_cid = self.cell_emb(cell_idx) * avail[:, 6:7]
        cell_repr = torch.cat([h_gene * avail[:, 0:1], h_scf * avail[:, 1:2],
                               h_mut, h_path, h_cid,
                               avail[:, 2:3], avail[:, 3:4]], dim=-1)

        h_fp = self.fp_enc(x_fp) * avail[:, 4:5]
        h_tgt = self.tgt_enc(tgt_idx, tgt_mask) * avail[:, 5:6]
        h_did = self.drug_emb(drug_idx) * avail[:, 7:8]
        drug_repr = torch.cat([h_fp, h_tgt, h_did, avail[:, 5:6]], dim=-1)

        x0 = self.input_proj(torch.cat([cell_repr, drug_repr], dim=-1))
        xc = x0
        for layer in self.cross_layers:
            xc = layer(x0, xc)
        xd = x0
        for layer in self.deep_layers:
            xd = layer(xd)
        return self.head(torch.cat([xc, xd], dim=-1)).squeeze(-1)
