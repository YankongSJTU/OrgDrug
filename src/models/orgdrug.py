# -*- coding: utf-8 -*-
"""
OrgDrug — Organoid Drug response engine
=========================================
Successor architecture to GDRNet (same multi-modal spirit, new interaction
engine). See OrgDrug/DESIGN.md.

Key differences vs GDRNet v1:
  * Set-valued views: mutation driver genes & drug targets are token sets over
    a SHARED gene embedding table (explicit drug-gene alignment).
  * Target-guided gating: drug target pooling gates the mutation/pathway views.
  * 8-token Transformer interaction (6 modality views + 2 ID tokens) with
    attention-based missing-modality masking (replaces v1's zero-vector hack).
  * Retains the winning DCNv2 low-rank explicit crossing + residual MLP.
  * Multi-objective loss: extreme-weighted Huber + in-cell ListNet ranking +
    auxiliary sensitivity classification.

Cell views : gene(2000) scf(3072) mut(set) path(51)
Drug views : fp(Morgan+MACCS+pathway-onehot) tgt(set)
ID tokens  : cell_id(64) drug_id(64)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

NEG_INF = -3e4  # fp16-safe mask value (|x| < 65504)


# ─── Building blocks ────────────────────────────────────────────────────────

class MLPView(nn.Module):
    """MLP view encoder. High-dim inputs (gene/scf/fp) use BatchNorm1d per
    feature (v1 recipe — per-sample LayerNorm rescales sparse fingerprints
    noisily); low-dim pathway scores use LayerNorm."""

    def __init__(self, in_dim, hidden, out_dim, dropout=0.15):
        super().__init__()
        norm = nn.BatchNorm1d(in_dim) if in_dim >= 512 else nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            norm,
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        nn.init.kaiming_normal_(self.net[1].weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.net[4].weight, nonlinearity="linear")

    def forward(self, x):
        return self.net(x)


class SetView(nn.Module):
    """Set-valued gene view: shared gene embedding table + attention pooling.

    idx  : (B, K) int64, gene vocab ids (1-based), 0 = none/pad
    mask : (B, K) bool, True = valid token
    Returns (token_out d_out, pooled_mid d_mid) — the latter drives gating.
    """

    def __init__(self, vocab_size, d_emb=32, d_mid=64, d_out=256,
                 n_heads=4, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, d_emb, padding_idx=0)
        nn.init.normal_(self.emb.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[0].zero_()
        self.proj = nn.Linear(d_emb, d_mid)
        self.query = nn.Parameter(torch.randn(1, 1, d_mid) * 0.02)
        self.attn = nn.MultiheadAttention(d_mid, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d_mid)
        self.out = nn.Linear(d_mid, d_out)

    def forward(self, idx, mask):
        e = self.proj(self.emb(idx))                       # B,K,mid
        key_pad = ~mask                                    # True = ignore
        empty = ~mask.any(dim=1, keepdim=True)             # B,1 fully padded
        # fully-empty rows: unmask position 0 (none-embedding is zero -> attn out 0)
        key_pad = key_pad & ~empty
        q = self.query.expand(idx.size(0), -1, -1)         # B,1,mid
        a, _ = self.attn(q, e, e, key_padding_mask=key_pad)
        mid = self.norm(a.squeeze(1))                      # B,mid
        return self.out(mid), mid


class CrossLayer(nn.Module):
    """Low-rank DCN v2 cross layer (identical math to GDRNet v1)."""

    def __init__(self, dim, rank=64):
        super().__init__()
        self.U = nn.Linear(dim, rank, bias=False)
        self.V = nn.Linear(rank, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, x0, xl):
        return x0 * self.V(F.relu(self.U(xl))) + self.bias + xl


class ResBlock(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.ln = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)
        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")

    def forward(self, x):
        return self.drop(F.gelu(self.ln(self.fc(x)))) + self.shortcut(x)


# ─── OrgDrug ────────────────────────────────────────────────────────────────

class OrgDrug(nn.Module):
    """
    avail : (B, 8) float, availability of
            [gene, scf, mut, path, fp, tgt, cell_id, drug_id]
    Ablation switches mirror GDRNet-style component isolation.
    """

    def __init__(
        self,
        n_genes=2000,
        scf_dim=3072,
        fp_dim=2235,           # 2048 Morgan + 167 MACCS + n_pathway_onehot
        path_dim=51,
        n_shared_genes=448,    # union(driver genes, target genes)
        n_cells=965,
        n_drugs=230,
        d_view=256,
        id_emb_dim=64,
        n_cross=2,
        cross_rank=64,
        n_attn_layers=2,
        dropout=0.15,
        # ablation switches
        use_transformer=True,
        use_tgca=True,
        use_cross=True,
        use_deep=True,
        id_bypass=False,
        use_cna=False,
        cna_dim=448,
    ):
        super().__init__()
        self.use_transformer = use_transformer
        self.use_tgca = use_tgca
        self.use_cross = use_cross
        self.use_deep = use_deep
        self.use_cna = use_cna

        # cell views
        self.gene_enc = MLPView(n_genes, 512, d_view, dropout)
        self.scf_enc = MLPView(scf_dim, 512, d_view, dropout)
        self.path_enc = MLPView(path_dim, 128, d_view, dropout)
        self.mut_view = SetView(n_shared_genes, d_out=d_view)
        if use_cna:
            # copy-number view over the same driver-gene panel (goal #1)
            self.cna_enc = MLPView(cna_dim, 128, d_view, dropout)

        # drug views
        self.fp_enc = MLPView(fp_dim, 512, d_view, dropout)
        self.tgt_view = SetView(n_shared_genes, d_out=d_view)

        # ID tokens
        self.cell_emb = nn.Embedding(n_cells + 1, id_emb_dim, padding_idx=0)
        self.drug_emb = nn.Embedding(n_drugs + 1, id_emb_dim, padding_idx=0)
        nn.init.normal_(self.cell_emb.weight, std=0.02)
        nn.init.normal_(self.drug_emb.weight, std=0.02)
        self.cell_tok = nn.Linear(id_emb_dim, d_view)
        self.drug_tok = nn.Linear(id_emb_dim, d_view)

        # target-guided gating (TGCA): drug target pooled mid -> sigmoid gate
        self.gate = nn.Linear(64, d_view)

        # token interaction
        n_slots = (7 if id_bypass else 9) if use_cna else (6 if id_bypass else 8)
        self.type_emb = nn.Parameter(torch.randn(1, n_slots, d_view) * 0.02)
        self.in_norm = nn.LayerNorm(d_view)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_view, nhead=4, dim_feedforward=512, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_attn_layers)

        d_pool = 2 * d_view

        # learned fusion over concatenated token slots (v1-style block-aligned
        # explicit crossing); missing slots zeroed before the projection.
        # id_bypass: ID tokens are excluded from attention/fusion entirely and
        # are concatenated afterwards (v1 recipe) so the interaction engine
        # cannot route memorization through the ID channel.
        self.id_bypass = id_bypass
        self.fusion = nn.Sequential(
            nn.Linear(n_slots * d_view + n_slots, d_pool),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool_norm = nn.LayerNorm(d_pool)
        if id_bypass:
            self.id_proj = nn.Sequential(
                nn.Linear(d_pool + 2 * id_emb_dim, d_pool),
                nn.GELU(),
            )

        # explicit crossing (retained from v1)
        if use_cross:
            self.cross_layers = nn.ModuleList(
                [CrossLayer(d_pool, rank=cross_rank) for _ in range(n_cross)])
        if use_deep:
            deep_cfg = [(d_pool, 256), (256, 128), (128, 64)]
            self.deep_layers = nn.ModuleList(
                [ResBlock(di, do, dropout) for di, do in deep_cfg])

        d_head_in = (d_pool if use_cross else 0) + (64 if use_deep else 0)
        if d_head_in == 0:
            d_head_in = d_pool
        self.head = nn.Sequential(
            nn.Linear(d_head_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reg_head = nn.Linear(256, 1)
        self.cls_head = nn.Linear(256, 1)

    def _masked_pool(self, tokens, avail):
        """avail (B,8) float 1/0 -> masked mean ⊕ max pool (B, 2*d)."""
        a = avail.unsqueeze(-1)                                  # B,8,1
        valid = (a > 0.5).float()
        n = valid.sum(dim=1).clamp(min=1.0)                      # B,1
        mean = (tokens * valid).sum(dim=1) / n                   # B,d
        tmax = tokens.masked_fill(valid == 0, NEG_INF).max(dim=1).values
        tmax = torch.nan_to_num(tmax, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.cat([mean, tmax], dim=-1)

    def forward(self, x_gene, scf, x_fp, path, mut_idx, mut_mask,
                tgt_idx, tgt_mask, cell_idx, drug_idx, avail, cna=None):
        # ── view tokens ──
        t_gene = self.gene_enc(x_gene)
        t_scf = self.scf_enc(scf)
        t_path = self.path_enc(path)
        t_mut, _ = self.mut_view(mut_idx, mut_mask)
        t_fp = self.fp_enc(x_fp)
        t_tgt, tgt_mid = self.tgt_view(tgt_idx, tgt_mask)

        # ── target-guided gating of mutation & pathway views ──
        if self.use_tgca:
            g = torch.sigmoid(self.gate(tgt_mid))               # B,d
            t_mut = t_mut * (0.5 + g)
            t_path = t_path * (0.5 + g)

        t_cid = self.cell_tok(self.cell_emb(cell_idx))
        t_did = self.drug_tok(self.drug_emb(drug_idx))

        if self.use_cna:
            # avail layout: [gene, scf, mut, path, cna, fp, tgt(, cid, did)]
            assert cna is not None, "use_cna model needs the cna view"
            t_cna = self.cna_enc(cna)
            if self.use_tgca:
                t_cna = t_cna * (0.5 + g)
            if self.id_bypass:
                slots = [t_gene, t_scf, t_mut, t_path, t_cna, t_fp, t_tgt]
                slot_avail = avail[:, :7]
            else:
                slots = [t_gene, t_scf, t_mut, t_path, t_cna, t_fp, t_tgt,
                         t_cid, t_did]
                slot_avail = avail
        elif self.id_bypass:
            slots = [t_gene, t_scf, t_mut, t_path, t_fp, t_tgt]
            slot_avail = avail[:, :6]
        else:
            slots = [t_gene, t_scf, t_mut, t_path, t_fp, t_tgt, t_cid, t_did]
            slot_avail = avail

        tokens = torch.stack(slots, dim=1)                        # B,S,d
        n_slots = tokens.size(1)
        tokens = self.in_norm(tokens + self.type_emb[:, :n_slots])

        if self.use_transformer:
            key_pad = slot_avail < 0.5                            # True = ignore
            tokens = self.encoder(tokens, src_key_padding_mask=key_pad)

        # block-aligned fusion: zero out missing slots, flatten, project
        tokens = tokens * (slot_avail.unsqueeze(-1) > 0.5).float()
        flat = torch.cat([tokens.flatten(1), slot_avail], dim=-1)
        pooled = self.pool_norm(self.fusion(flat))                # B, 2d
        if self.id_bypass:
            # v1 recipe: ID embeddings bypass attention, concatenated into the
            # base vector that feeds the explicit cross network
            id_vec = torch.cat([self.cell_emb(cell_idx),
                                self.drug_emb(drug_idx)], dim=-1)
            pooled = self.id_proj(torch.cat([pooled, id_vec], dim=-1))

        # ── explicit crossing + deep (retained winning components) ──
        outs = []
        if self.use_cross:
            xc = pooled
            for layer in self.cross_layers:
                xc = layer(pooled, xc)
            outs.append(xc)
        if self.use_deep:
            xd = pooled
            for layer in self.deep_layers:
                xd = layer(xd)
            outs.append(xd)
        if not outs:
            outs.append(pooled)
        h = self.head(torch.cat(outs, dim=-1))

        return self.reg_head(h).squeeze(-1), self.cls_head(h).squeeze(-1)


# ─── Dataset ────────────────────────────────────────────────────────────────

class OrgDrugDataset(Dataset):
    """Entity-level feature storage; pairs reference rows by index.

    cell_feats / drug_feats are dicts of numpy arrays (row-aligned).
    """

    def __init__(self, cell_feats, drug_feats, cell_idx, drug_idx, y,
                 w_ext=None, sens_label=None, avail=None,
                 emb_cell_idx=None, emb_drug_idx=None):
        self.gene = torch.FloatTensor(cell_feats["gene"])
        self.scf = torch.FloatTensor(cell_feats["scf"])
        self.path = torch.FloatTensor(cell_feats["path"])
        self.mut_idx = torch.LongTensor(cell_feats["mut_idx"])
        self.mut_mask = torch.BoolTensor(cell_feats["mut_mask"])
        self.fp = torch.FloatTensor(drug_feats["fp"])
        self.tgt_idx = torch.LongTensor(drug_feats["tgt_idx"])
        self.tgt_mask = torch.BoolTensor(drug_feats["tgt_mask"])
        self.cell_idx = torch.LongTensor(cell_idx)
        self.drug_idx = torch.LongTensor(drug_idx)
        # embedding-table indices may differ from feature-row indices
        # (organoids: cell -> 0/none; unseen drugs -> 0)
        self.emb_ci = torch.LongTensor(emb_cell_idx) if emb_cell_idx is not None else self.cell_idx
        self.emb_di = torch.LongTensor(emb_drug_idx) if emb_drug_idx is not None else self.drug_idx
        self.y = torch.FloatTensor(y)
        self.w_ext = torch.FloatTensor(w_ext) if w_ext is not None else torch.ones(len(y))
        self.sens = torch.FloatTensor(sens_label) if sens_label is not None else torch.zeros(len(y))
        self.avail = torch.FloatTensor(avail) if avail is not None else torch.ones(len(y), 8)
        # optional copy-number view (rows aligned with cell features)
        self.cna = torch.FloatTensor(cell_feats["cna"]) if "cna" in cell_feats else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        items = (self.gene[self.cell_idx[i]], self.scf[self.cell_idx[i]],
                 self.fp[self.drug_idx[i]], self.path[self.cell_idx[i]],
                 self.mut_idx[self.cell_idx[i]], self.mut_mask[self.cell_idx[i]],
                 self.tgt_idx[self.drug_idx[i]], self.tgt_mask[self.drug_idx[i]],
                 self.emb_ci[i], self.emb_di[i], self.avail[i],
                 self.y[i], self.w_ext[i], self.sens[i])
        if self.cna is not None:
            items = items + (self.cna[self.cell_idx[i]],)
        return items


# ─── Losses ─────────────────────────────────────────────────────────────────

def masked_softmax_rows(x, same, tau):
    """softmax over same-group entries per row. x (B,B)/tau, same (B,B) bool."""
    z = torch.where(same, x / tau, torch.full_like(x, NEG_INF))
    return F.softmax(z, dim=1)


def listnet_loss(pred, y, cell_idx, tau=2.5, min_group=3):
    """In-cell ListNet: CE between true/pred within-cell softmax distributions."""
    pred = pred.float()
    y = y.float()
    same = cell_idx.unsqueeze(0) == cell_idx.unsqueeze(1)        # B,B
    cnt = same.sum(dim=1)
    valid = cnt >= min_group
    if valid.sum().item() < min_group:
        return pred.new_tensor(0.0)
    yb = y.unsqueeze(1).expand(-1, y.size(0))                    # B,B : yb[i,j]=y[j]
    with torch.no_grad():
        p = masked_softmax_rows(yb, same, tau)
    logq = torch.log(masked_softmax_rows(pred.unsqueeze(1).expand(-1, pred.size(0)),
                                         same, tau) + 1e-9)
    ce = -(p * logq).sum(dim=1)                                 # B
    return ce[valid].mean()


def orgdrug_loss(reg, cls, y, w_ext, sens, cell_idx,
                 lambda_list=0.1, lambda_cls=0.2,
                 use_list=True, use_cls=True, use_extw=True):
    per = F.huber_loss(reg, y, delta=1.0, reduction="none")
    if use_extw:
        per = per * w_ext
    loss = per.mean()
    if use_list:
        loss = loss + lambda_list * listnet_loss(reg, y, cell_idx)
    if use_cls:
        loss = loss + lambda_cls * F.binary_cross_entropy_with_logits(cls, sens)
    return loss


# ─── Unseen-entity embedding fix (LCLO protocol, mirrors v1) ───────────────

def fix_unseen_embeddings(model, tr_cell_idx):
    raw = model.module if hasattr(model, "module") else model
    if hasattr(raw, "m"):            # _Wrapped(OrgDrugClassic)
        raw = raw.m
    tr_cells = np.unique(tr_cell_idx)
    all_cells = np.arange(raw.cell_emb.num_embeddings)
    unseen = np.setdiff1d(all_cells, tr_cells)
    if len(unseen) == 0:
        return
    with torch.no_grad():
        trained_emb = raw.cell_emb.weight[tr_cells].mean(dim=0)
        for idx in unseen:
            raw.cell_emb.weight[idx] = trained_emb.clone()
