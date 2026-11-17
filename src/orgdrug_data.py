# -*- coding: utf-8 -*-
"""
OrgDrug shared data helpers
============================
Feature-table loading + entity-row assembly used by pretraining and organoid
fine-tuning (the LCLO benchmark script keeps its own copy for independence).
"""

import os
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ORGDRUG_ROOT",
                 Path(__file__).resolve().parents[1]))
FEAT = ROOT / "data/processed/orgdrug"
GEO_PROC = ROOT / "data/processed/geo"
PROC = ROOT / "data/processed"
RAW_GEO = ROOT / "data/raw/geo"

K_MUT, K_TGT = 64, 16


def load_tables():
    """Load all OrgDrug feature tables (mutation / pathway / targets / MACCS)."""
    t = {}
    t["path_mat"] = np.load(FEAT / "pathway_scores.npy")
    t["path_ids"] = np.load(FEAT / "pathway_cell_ids.npy", allow_pickle=True)
    t["path_dim"] = t["path_mat"].shape[1]

    t["mut_mat"] = np.load(FEAT / "mut_matrix.npy")
    t["mut_ids"] = np.load(FEAT / "mut_cell_ids.npy", allow_pickle=True)
    mg = pd.read_csv(FEAT / "mut_genes.csv")
    t["mut_genes"] = mg["gene"].values
    # curated-first ordering so truncation keeps canonical drivers
    t["mut_order"] = np.argsort(~mg["curated"].values)
    t["mut_genes_ordered"] = t["mut_genes"][t["mut_order"]]

    t["vocab"] = pd.read_csv(FEAT / "shared_gene_vocab.csv")
    t["gene2idx"] = dict(zip(t["vocab"]["gene"], t["vocab"]["vocab_idx"]))
    t["n_vocab"] = len(t["vocab"])

    t["maccs"] = np.load(FEAT / "maccs_matrix.npy")
    t["maccs_names"] = pd.read_csv(FEAT / "maccs_drugs.csv")["drug_name"].values
    t["onehot"] = np.load(FEAT / "drug_pathway_onehot.npy")
    t["onehot_names"] = pd.read_csv(FEAT / "drug_target_drugs.csv")["drug_name"].values
    t["tgt_mat"] = np.load(FEAT / "drug_target_matrix.npy")
    t["tgt_genes"] = pd.read_csv(FEAT / "target_genes.csv")["gene"].values
    t["fp_dim"] = 2048 + t["maccs"].shape[1] + t["onehot"].shape[1]
    return t


def cell_set_rows(t, model_ids):
    """Per-cell rows: pathway scores, mutation set indices."""
    ids = list(model_ids)
    path_row = {c: i for i, c in enumerate(t["path_ids"])}
    mut_row = {c: i for i, c in enumerate(t["mut_ids"])}

    path = np.stack([
        t["path_mat"][path_row[c]] if c in path_row else np.zeros(t["path_dim"], np.float32)
        for c in ids])

    mut_idx = np.zeros((len(ids), K_MUT), dtype=np.int64)
    mut_mask = np.zeros((len(ids), K_MUT), dtype=bool)
    for ci, c in enumerate(ids):
        if c not in mut_row:
            continue
        row = t["mut_mat"][mut_row[c]]
        hit = t["mut_genes_ordered"][row[t["mut_order"]] > 0]
        idxs = [t["gene2idx"][g] for g in hit if g in t["gene2idx"]][:K_MUT]
        mut_idx[ci, :len(idxs)] = idxs
        mut_mask[ci, :len(idxs)] = len(idxs) > 0
    return path, mut_idx, mut_mask


def drug_set_rows(t, drug_names, morgan_by_drug):
    """Per-drug rows: fp (morgan+MACCS+onehot), target set indices."""
    names = list(drug_names)
    maccs_row = {d: i for i, d in enumerate(t["maccs_names"])}
    onehot_row = {d: i for i, d in enumerate(t["onehot_names"])}

    fp = np.zeros((len(names), t["fp_dim"]), dtype=np.float32)
    for di, d in enumerate(names):
        if d in morgan_by_drug:
            fp[di, :2048] = morgan_by_drug[d]
        if d in maccs_row:
            m = t["maccs"][maccs_row[d]]
            fp[di, 2048:2048 + len(m)] = m
        if d in onehot_row:
            fp[di, 2048 + t["maccs"].shape[1]:] = t["onehot"][onehot_row[d]]

    tgt_idx = np.zeros((len(names), K_TGT), dtype=np.int64)
    tgt_mask = np.zeros((len(names), K_TGT), dtype=bool)
    for di, d in enumerate(names):
        if d not in onehot_row:
            continue
        hit = t["tgt_genes"][t["tgt_mat"][onehot_row[d]] > 0]
        idxs = [t["gene2idx"][g] for g in hit if g in t["gene2idx"]][:K_TGT]
        tgt_idx[di, :len(idxs)] = idxs
        tgt_mask[di, :len(idxs)] = len(idxs) > 0
    return fp, tgt_idx, tgt_mask


# ─── Per-drug train statistics (for extreme weighting & sens labels) ───────

def drug_train_stats(y, drug_names, tr_mask):
    df = pd.DataFrame({"y": y[tr_mask], "d": drug_names[tr_mask]})
    stats = df.groupby("d")["y"].agg(["mean", "std", lambda s: np.percentile(s, 30)])
    stats.columns = ["mean", "std", "q30"]
    stats["std"] = stats["std"].replace(0, 1.0)
    gm, gs = df["y"].mean(), max(df["y"].std(), 1e-6)
    return stats, gm, gs


def compute_pair_weights(y, drug_names, stats, gm, gs):
    m = stats.reindex(drug_names)
    mean = m["mean"].fillna(gm).values
    std = m["std"].fillna(gs).values
    q30 = m["q30"].fillna(np.percentile(y, 30)).values
    z = (y - mean) / std
    w = 1.0 + 0.5 * np.clip(np.abs(z), 0, 3) / 3
    sens = (y <= q30).astype(np.float32)
    return w.astype(np.float32), sens


# ─── Organoid expression -> pathway / landmark features ─────────────────────

def _zscore(df):
    mu = df.mean(axis=0)
    sd = df.std(axis=0).replace(0, 1.0)
    return (df - mu) / sd


def _as_genes_by_samples(df, sample_hint):
    """Normalize a GEO expression frame to genes(index) x samples(columns)."""
    cols_like_samples = sum(sample_hint in str(c) for c in df.columns)
    idx_like_samples = sum(sample_hint in str(i) for i in df.index)
    if idx_like_samples > cols_like_samples and idx_like_samples > 3:
        df = df.T
    return df


def organoid_expression_features(cancer, organoid_ids):
    """Compute (landmark_z, pathway_scores) for organoids from raw GEO matrices.

    Returns (landmark (n,2000) or None, pathway (n,path_dim) or None, path_dim).
    CRC has no usable cross-platform landmarks -> (None, None, path_dim).
    """
    genesets = {k: sorted(v.keys()) for k, v in
                json.loads((FEAT / "hallmark_genesets.json").read_text())
                ["MSigDB_Hallmark_2020"]["terms"].items()}
    stem = json.loads((FEAT / "stemness_panel.json").read_text())
    landmark = list(pd.read_parquet(PROC / "gdsc_cell_features.parquet").columns)
    terms = {"HALLMARK_" + k: v for k, v in genesets.items()}
    terms["MICRO_Stemness"] = stem
    path_dim = len(terms)

    if cancer == "pdac":
        with gzip.open(RAW_GEO / "GSE194249/GSE194249_PDPCOs_FPKM.txt.gz", "rt") as f:
            expr = pd.read_csv(f, sep="\t", index_col=0)
        expr = _as_genes_by_samples(expr, "CAS-")
        expr = expr.loc[:, expr.columns.str.startswith("CAS-DAC")]
        print(f"    pdac expression: {expr.shape[0]} genes x {expr.shape[1]} DAC samples", flush=True)
    elif cancer == "blca":
        with gzip.open(RAW_GEO / "GSE103990/GSE103990_Normalized_counts.txt.gz", "rt") as f:
            expr = pd.read_csv(f, sep="\t", index_col=0)
        expr.index = expr.index.astype(str).str.split("_", n=1).str[-1]
        expr = expr[~expr.index.duplicated()]
        expr = _as_genes_by_samples(expr, "MS")
        print(f"    blca expression: {expr.shape[0]} genes x {expr.shape[1]} samples", flush=True)
    else:
        return None, None, path_dim

    keep = [c for c in organoid_ids if c in expr.columns]
    if len(keep) < max(3, len(organoid_ids) // 2):
        return None, None, path_dim
    expr = expr[keep].T.apply(lambda s: np.log2(s.astype(float) + 1.0))
    expr = _zscore(expr)

    lm = np.zeros((len(organoid_ids), 2000), dtype=np.float32)
    pw = np.zeros((len(organoid_ids), path_dim), dtype=np.float32)
    pos = {o: i for i, o in enumerate(organoid_ids)}
    common_lm = [g for g in landmark if g in expr.columns]
    for o in keep:
        lm[pos[o]] = _fill_landmarks(expr.loc[o], landmark)
    for j, (name, genes) in enumerate(terms.items()):
        present = [g for g in genes if g in expr.columns]
        if len(present) < 5:
            continue
        vals = expr[present].mean(axis=1)
        for o in keep:
            pw[pos[o], j] = vals.loc[o]
    has_lm = len(common_lm) >= 1800
    return (lm if has_lm else None), pw, path_dim


def _fill_landmarks(row, landmark):
    out = np.zeros(2000, dtype=np.float32)
    for i, g in enumerate(landmark):
        if g in row.index:
            out[i] = row[g]
    return out


def load_organoid_cohort(cancer):
    """Load a prepared organoid cohort (scF emb, Morgan FP per pair, response)."""
    if cancer == "crc":
        meta = pd.read_csv(PROC / "organoid_pair_meta.csv")
        emb = np.load(PROC / "organoid_cell_emb.npy")
        ids = np.load(PROC / "organoid_cell_ids.npy", allow_pickle=True)
        fp = np.load(PROC / "organoid_drug_features.npy")
        y = np.load(PROC / "organoid_response.npy")
    else:
        meta = pd.read_csv(GEO_PROC / f"{cancer}_pair_meta.csv")
        emb = np.load(GEO_PROC / f"{cancer}_cell_emb.npy")
        ids = np.load(GEO_PROC / f"{cancer}_cell_ids.npy", allow_pickle=True)
        fp = np.load(GEO_PROC / f"{cancer}_drug_features.npy")
        y = np.load(GEO_PROC / f"{cancer}_response.npy")
    emb_map = {o: emb[i] for i, o in enumerate(ids)}
    scf = np.stack([emb_map[o] for o in meta["organoid_id"]]).astype(np.float32)
    fp = fp[:, :2048].astype(np.float32)
    return meta, scf, fp, y.astype(np.float32)
