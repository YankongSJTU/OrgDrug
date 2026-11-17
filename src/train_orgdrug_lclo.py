# -*- coding: utf-8 -*-
"""
OrgDrug LCLO Benchmark (E1 head-to-head / E2 ablations / E3 mutation analysis)
================================================================================
Trains OrgDrug under the *identical* GDSC2 leave-cell-line-out protocol as
GDRNet v1 (same folds via load_gdsc_lclo(seed=42), same metrics, same early
stopping, same unseen-cell embedding fix), then:

  * saves per-fold test predictions (parquet) + fold evals (json)
  * aggregates ensemble metrics and compares against GDRNet's saved
    predictions with a paired Wilcoxon test on per-drug r
  * mutation-stratified MAE analysis (KRAS / TP53 / BRAF, v1 Table S3 protocol)

Usage:
  python train_orgdrug_lclo.py --device cuda:6 --variant full
  python train_orgdrug_lclo.py --device cuda:6 --variant no-mut --seeds 42
  python train_orgdrug_lclo.py --device cuda:6 --variant full --folds 0   # smoke
"""

import os
import argparse
import io
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("ORGDRUG_ROOT",
                 Path(__file__).resolve().parents[1]))
FEAT = ROOT / "data/processed/orgdrug"
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass

from data_loader import load_gdsc_lclo                     # noqa: E402
from eval_utils import compute_full_eval                   # noqa: E402
from models.orgdrug import (OrgDrug, OrgDrugDataset,       # noqa: E402
                            orgdrug_loss, fix_unseen_embeddings)

K_MUT, K_TGT = 64, 16
VARIANTS = {
    "full":           dict(),
    "no-mut":         dict(drop_mut=True),
    "no-path":        dict(drop_path=True),
    "no-tgt":         dict(drop_tgt=True),
    "no-tgca":        dict(use_tgca=False),
    "no-transformer": dict(use_transformer=False),
    "no-cross":       dict(use_cross=False),
    "no-list":        dict(use_list=False),
    "no-extw":        dict(use_extw=False),
    "no-cls":         dict(use_cls=False),
    "no-id":          dict(drop_id=True),
    "classic":        dict(classic=True),
    "classic-pure":   dict(classic=True, use_extw=False),
    "cna":            dict(use_cna=True),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="full", choices=list(VARIANTS))
    p.add_argument("--device", default="cuda:6")
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--view-dropout", type=float, default=0.08)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lambda-list", type=float, default=0.1)
    p.add_argument("--lambda-cls", type=float, default=0.2)
    p.add_argument("--n-cross", type=int, default=2)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out", default=None)
    p.add_argument("--init-ckpt", default=None,
                   help="warm-start from pretrained checkpoint(s), comma list "
                        "aligned with --seeds; shape-filtered, ID embeddings "
                        "excluded (LCLO entity indices differ)")
    return p.parse_args()


args = parse_args()
VAR = VARIANTS[args.variant]
OUT = ROOT / (args.out or f"results/orgdrug/lclo_{args.variant}")
OUT.mkdir(parents=True, exist_ok=True)


# ─── Feature assembly ───────────────────────────────────────────────────────

def build_entity_features(data):
    """Map OrgDrug features onto the LCLO entity lists (cells / drugs)."""
    meta = data.meta
    cell_list, drug_list = data.cell_list, data.drug_list

    # per-cell first-occurrence row of the per-pair matrices
    first_cell = pd.Series(np.arange(len(meta)), index=meta["ModelID"].values).groupby(level=0).first()
    gene_by_cell = data.X_gene[first_cell.reindex(cell_list).values].astype(np.float32)
    scf_by_cell = data.X_scf[first_cell.reindex(cell_list).values].astype(np.float32)

    # pathway scores
    path_mat = np.load(FEAT / "pathway_scores.npy")
    path_ids = np.load(FEAT / "pathway_cell_ids.npy", allow_pickle=True)
    path_row = {c: i for i, c in enumerate(path_ids)}
    n_path = path_mat.shape[1]
    path_by_cell = np.stack([
        path_mat[path_row[c]] if c in path_row else np.zeros(n_path, np.float32)
        for c in cell_list])

    # shared gene vocab
    vocab = pd.read_csv(FEAT / "shared_gene_vocab.csv")
    gene2idx = dict(zip(vocab["gene"], vocab["vocab_idx"]))  # 1-based
    n_vocab = len(vocab)

    # mutation set indices (curated genes first so truncation keeps drivers)
    mut_genes_df = pd.read_csv(FEAT / "mut_genes.csv")
    mut_mat = np.load(FEAT / "mut_matrix.npy")
    mut_ids = np.load(FEAT / "mut_cell_ids.npy", allow_pickle=True)
    mut_row = {c: i for i, c in enumerate(mut_ids)}
    order = np.argsort(~mut_genes_df["curated"].values)  # curated first
    mut_genes_ordered = mut_genes_df["gene"].values[order]

    mut_idx = np.zeros((len(cell_list), K_MUT), dtype=np.int64)
    mut_mask = np.zeros((len(cell_list), K_MUT), dtype=bool)
    for ci, c in enumerate(cell_list):
        if c not in mut_row:
            continue
        row = mut_mat[mut_row[c]]
        hit = mut_genes_ordered[row[order] > 0]
        idxs = [gene2idx[g] for g in hit if g in gene2idx][:K_MUT]
        mut_idx[ci, :len(idxs)] = idxs
        mut_mask[ci, :len(idxs)] = True

    # drug features: Morgan (first occurrence) + MACCS + pathway one-hot
    first_drug = pd.Series(np.arange(len(meta)), index=meta["drug_name"].values).groupby(level=0).first()
    morgan_by_drug = data.X_fp[first_drug.reindex(drug_list).values].astype(np.float32)

    maccs = np.load(FEAT / "maccs_matrix.npy")
    maccs_names = pd.read_csv(FEAT / "maccs_drugs.csv")["drug_name"].values
    maccs_row = {d: i for i, d in enumerate(maccs_names)}
    onehot = np.load(FEAT / "drug_pathway_onehot.npy")
    onehot_names = pd.read_csv(FEAT / "drug_target_drugs.csv")["drug_name"].values
    onehot_row = {d: i for i, d in enumerate(onehot_names)}

    n_oh = onehot.shape[1]
    fp_dim = 2048 + 167 + n_oh
    fp_by_drug = np.zeros((len(drug_list), fp_dim), dtype=np.float32)
    for di, d in enumerate(drug_list):
        fp_by_drug[di, :2048] = morgan_by_drug[di]
        if d in maccs_row:
            m = maccs[maccs_row[d]]
            fp_by_drug[di, 2048:2048 + len(m)] = m
        if d in onehot_row:
            fp_by_drug[di, 2048 + 167:] = onehot[onehot_row[d]]

    # target set indices
    tgt_mat = np.load(FEAT / "drug_target_matrix.npy")
    tgt_genes = pd.read_csv(FEAT / "target_genes.csv")["gene"].values
    tgt_idx = np.zeros((len(drug_list), K_TGT), dtype=np.int64)
    tgt_mask = np.zeros((len(drug_list), K_TGT), dtype=bool)
    for di, d in enumerate(drug_list):
        if d not in onehot_row:
            continue
        hit = tgt_genes[tgt_mat[onehot_row[d]] > 0]
        idxs = [gene2idx[g] for g in hit if g in gene2idx][:K_TGT]
        tgt_idx[di, :len(idxs)] = idxs
        tgt_mask[di, :len(idxs)] = len(idxs) > 0

    cell_feats = dict(gene=gene_by_cell, scf=scf_by_cell, path=path_by_cell,
                      mut_idx=mut_idx, mut_mask=mut_mask)
    drug_feats = dict(fp=fp_by_drug, tgt_idx=tgt_idx, tgt_mask=tgt_mask)

    # copy-number view over the driver panel (optional, variant "cna")
    if (FEAT / "cna_matrix.npy").exists():
        cna_mat = np.load(FEAT / "cna_matrix.npy")
        cna_ids = np.load(FEAT / "cna_cell_ids.npy", allow_pickle=True)
        cna_row = {c: i for i, c in enumerate(cna_ids)}
        cna_by_cell = np.stack([
            cna_mat[cna_row[c]] if c in cna_row else np.zeros(cna_mat.shape[1], np.float32)
            for c in cell_list])
        cna_avail = np.array([1.0 if c in cna_row else 0.0 for c in cell_list],
                             dtype=np.float32)
        cell_feats["cna"] = cna_by_cell.astype(np.float32)
        cell_feats["cna_avail"] = cna_avail
        print(f"  cna view ready: {cna_mat.shape[1]} genes, "
              f"{(cna_avail > 0).mean()*100:.0f}% cells covered", flush=True)

    print(f"  entities: {len(cell_list)} cells, {len(drug_list)} drugs; "
          f"path={n_path}, fp_dim={fp_dim}, vocab={n_vocab}", flush=True)
    print(f"  mut: mean {mut_mask.sum(1).mean():.1f} genes/cell "
          f"({(mut_mask.any(1)).mean()*100:.0f}% cells with >=1 driver mut); "
          f"tgt: {(tgt_mask.any(1)).mean()*100:.0f}% drugs annotated", flush=True)
    return cell_feats, drug_feats, n_path, fp_dim, n_vocab, gene2idx


def apply_variant_avail(avail, var):
    """Permanently disable views for ablation variants."""
    if var.get("drop_mut"):
        avail[:, 2] = 0.0
    if var.get("drop_path"):
        avail[:, 3] = 0.0
    if var.get("drop_tgt"):
        avail[:, 5] = 0.0
    if var.get("drop_id"):
        avail[:, 6] = 0.0
        avail[:, 7] = 0.0
    return avail


def build_pair_avail(data, cell_feats, drug_feats, n_cells):
    """Per-pair availability: all on except drugs without target annotation."""
    n = len(data.y)
    avail = np.ones((n, 8), dtype=np.float32)
    cell_has_mut = cell_feats["mut_mask"].any(axis=1)
    cell_has_path = np.abs(cell_feats["path"]).sum(axis=1) > 0
    drug_has_tgt = drug_feats["tgt_mask"].any(axis=1)
    ci = data.cell_idx - 1
    di = data.drug_idx - 1
    avail[:, 2] = cell_has_mut[ci]
    avail[:, 3] = cell_has_path[ci]
    avail[:, 5] = drug_has_tgt[di]
    return avail


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


# ─── Training ───────────────────────────────────────────────────────────────

def train_one(data, cell_feats, drug_feats, fp_dim, n_vocab, path_dim,
              tr_mask, te_mask, seed, device, var, init_ckpt=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    from torch.utils.data import DataLoader

    if var.get("classic"):
        from models.orgdrug_classic import OrgDrugClassic
        model = OrgDrugClassic(
            fp_dim=fp_dim, path_dim=path_dim, n_shared_genes=n_vocab,
            n_cells=data.n_cells, n_drugs=data.n_drugs,
            n_cross=args.n_cross, dropout=args.dropout).to(device)

        class _Wrapped(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, *b):
                return self.m(*b), None

        model = _Wrapped(model)
    else:
        model = OrgDrug(
            n_genes=cell_feats["gene"].shape[1],
            scf_dim=cell_feats["scf"].shape[1],
            fp_dim=fp_dim,
            path_dim=path_dim,
            n_shared_genes=n_vocab,
            n_cells=data.n_cells,
            n_drugs=data.n_drugs,
            dropout=args.dropout,
            n_cross=args.n_cross,
            use_transformer=var.get("use_transformer", True),
            use_tgca=var.get("use_tgca", True),
            use_cross=var.get("use_cross", True),
            use_deep=var.get("use_deep", True),
            use_cna=var.get("use_cna", False),
            cna_dim=(cell_feats["cna"].shape[1] if "cna" in cell_feats else 448),
        ).to(device)

    if init_ckpt:
        sd = torch.load(init_ckpt, map_location="cpu")
        own = model.state_dict()
        filtered = {k: v for k, v in sd.items()
                    if k in own and own[k].shape == v.shape
                    and "cell_emb" not in k and "drug_emb" not in k}
        own.update(filtered)
        model.load_state_dict(own)
        print(f" [init {len(filtered)}/{len(sd)} tensors <- {Path(init_ckpt).name}]",
              end="", flush=True)

    stats, gm, gs = drug_train_stats(data.y, data.drug_names, tr_mask)
    w_tr, sens_tr = compute_pair_weights(data.y[tr_mask], data.drug_names[tr_mask], stats, gm, gs)
    w_te, sens_te = compute_pair_weights(data.y[te_mask], data.drug_names[te_mask], stats, gm, gs)

    avail_all = build_pair_avail(data, cell_feats, drug_feats, data.n_cells)
    avail_all = apply_variant_avail(avail_all, var)
    use_cna = bool(var.get("use_cna")) and "cna" in cell_feats
    cf = dict(cell_feats)
    if use_cna:
        # insert cna availability at slot 4 (after path, before fp)
        cna_avail = cell_feats["cna_avail"][data.cell_idx - 1]
        avail_all = np.insert(avail_all, 4, cna_avail, axis=1)
    else:
        cf.pop("cna", None)          # Dataset appends cna items only when present
        cf.pop("cna_avail", None)

    tr_ds = OrgDrugDataset(cf, drug_feats, data.cell_idx[tr_mask] - 1,
                           data.drug_idx[tr_mask] - 1, data.y[tr_mask],
                           w_tr, sens_tr, avail_all[tr_mask],
                           emb_cell_idx=data.cell_idx[tr_mask],
                           emb_drug_idx=data.drug_idx[tr_mask])
    te_ds = OrgDrugDataset(cf, drug_feats, data.cell_idx[te_mask] - 1,
                           data.drug_idx[te_mask] - 1, data.y[te_mask],
                           w_te, sens_te, avail_all[te_mask],
                           emb_cell_idx=data.cell_idx[te_mask],
                           emb_drug_idx=data.drug_idx[te_mask])

    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                           num_workers=args.workers, pin_memory=True,
                           persistent_workers=args.workers > 0)
    te_loader = DataLoader(te_ds, batch_size=args.batch * 2, shuffle=False,
                           num_workers=2, pin_memory=True,
                           persistent_workers=args.workers > 0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = args.epochs * len(tr_loader)
    warmup = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    loss_kw = dict(
        use_list=var.get("use_list", True) and args.lambda_list > 0,
        use_cls=var.get("use_cls", True) and args.lambda_cls > 0
                 and not var.get("classic"),
        use_extw=var.get("use_extw", True),
        lambda_list=args.lambda_list,
        lambda_cls=args.lambda_cls,
    )

    best_rmse, best_state, wait = float("inf"), None, 0
    drop_views = [0, 1, 2, 3, 4, 5]

    for epoch in range(args.epochs):
        model.train()
        for items in tr_loader:
            items = list(items)
            cn = items.pop().to(device, non_blocking=True) if use_cna else None
            (xg, scf, fp, path, mi, mm, ti, tm, ci, di, av, yb, wb, sb) = items
            batch = [t.to(device, non_blocking=True) for t in
                     (xg, scf, fp, path, mi, mm, ti, tm, ci, di)]
            av = av.to(device)
            yb, wb, sb = yb.to(device), wb.to(device), sb.to(device)

            # random view dropout for missing-modality robustness
            if args.view_dropout > 0:
                drop = torch.rand(av.shape, device=device) < args.view_dropout
                drop[:, av.shape[1] - 2:] = False         # never drop ID tokens
                drop = drop & (av > 0.5)                  # only drop available views
                av = av * (~drop).float()

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
                reg, cls = model(*batch, av, cna=cn)
                loss = orgdrug_loss(reg, cls, yb, wb, sb, batch[8], **loss_kw)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # epoch eval on test fold (same protocol as v1)
        model.eval()
        preds = []
        with torch.no_grad():
            for raw_items in te_loader:
                items = list(raw_items)
                cn = items.pop().to(device, non_blocking=True) if use_cna else None
                (xg, scf, fp, path, mi, mm, ti, tm, ci, di, av, yb, wb, sb) = items
                batch = [t.to(device, non_blocking=True) for t in
                         (xg, scf, fp, path, mi, mm, ti, tm, ci, di)]
                reg, _ = model(*batch, av.to(device), cna=cn)
                preds.append(reg.float().cpu())
        preds = torch.cat(preds).numpy()
        rmse = float(np.sqrt(np.mean((data.y[te_mask] - preds) ** 2)))
        if rmse < best_rmse:
            best_rmse, wait = rmse, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
        if args.log_every and ((epoch + 1) % args.log_every == 0 or epoch == 0):
            r = float(np.corrcoef(data.y[te_mask], preds)[0, 1])
            print(f"      ep{epoch+1:3d}  rmse={rmse:.4f} r={r:.4f} "
                  f"wait={wait}", flush=True)

    model.load_state_dict(best_state)
    model = model.to(device)
    fix_unseen_embeddings(model, data.cell_idx[tr_mask])

    model.eval()
    preds = []
    with torch.no_grad():
        for raw_items in te_loader:
            items = list(raw_items)
            cn = items.pop().to(device, non_blocking=True) if use_cna else None
            (xg, scf, fp, path, mi, mm, ti, tm, ci, di, av, yb, wb, sb) = items
            batch = [t.to(device, non_blocking=True) for t in
                     (xg, scf, fp, path, mi, mm, ti, tm, ci, di)]
            reg, _ = model(*batch, av.to(device), cna=cn)
            preds.append(reg.float().cpu())
    return np.concatenate(preds).astype(np.float32)


# ─── Analysis ───────────────────────────────────────────────────────────────

def compare_with_gdrnet(OUT, fold_list):
    """Paired per-drug r comparison vs GDRNet saved predictions."""
    ref_dir = ROOT / "results/tables/lclo_rerun"
    if not ref_dir.exists():
        return None
    from scipy.stats import wilcoxon
    ours_r, theirs_r, labels = [], [], []
    for f in fold_list:
        try:
            g = pd.read_parquet(ref_dir / f"fold{f}_predictions.parquet")
            o = pd.read_parquet(OUT / f"fold{f}_predictions.parquet")
        except Exception:
            continue
        m = o.merge(g[["cell_id", "drug_name", "y_true", "GDRNet-Ensemble"]],
                    on=["cell_id", "drug_name"], suffixes=("", "_g"))
        for d, sub in m.groupby("drug_name"):
            if len(sub) < 5:
                continue
            ro = np.corrcoef(sub["y_true"], sub["OrgDrug-Ensemble"])[0, 1]
            rg = np.corrcoef(sub["y_true_g"], sub["GDRNet-Ensemble"])[0, 1]
            if np.isfinite(ro) and np.isfinite(rg):
                ours_r.append(ro)
                theirs_r.append(rg)
                labels.append(f"{d}|f{f}")
    if len(ours_r) < 10:
        return None
    ours_r, theirs_r = np.array(ours_r), np.array(theirs_r)
    stat, p = wilcoxon(ours_r, theirs_r)
    return dict(
        n_pairs=len(ours_r),
        orgdrug_perdrug=float(ours_r.mean()),
        gdrnet_perdrug=float(theirs_r.mean()),
        delta=float((ours_r - theirs_r).mean()),
        wilcoxon_p=float(p),
        n_improved=int((ours_r > theirs_r).sum()),
    )


def mutation_stratified(OUT, fold_list, gene2idx):
    """MAE by KRAS/TP53/BRAF mutation status (v1 Table S3 protocol)."""
    from scipy.stats import mannwhitneyu
    mut_mat = np.load(FEAT / "mut_matrix.npy")
    mut_genes = pd.read_csv(FEAT / "mut_genes.csv")["gene"].tolist()
    mut_ids = set(np.load(FEAT / "mut_cell_ids.npy", allow_pickle=True).tolist())
    status = {}
    for gene in ["KRAS", "TP53", "BRAF"]:
        if gene in mut_genes:
            col = mut_mat[:, mut_genes.index(gene)]
            status[gene] = dict(zip(np.load(FEAT / "mut_cell_ids.npy", allow_pickle=True), col > 0))

    rows = []
    for f in fold_list:
        fp_ = OUT / f"fold{f}_predictions.parquet"
        if not fp_.exists():
            continue
        o = pd.read_parquet(fp_)
        o["abs_err"] = (o["y_true"] - o["OrgDrug-Ensemble"]).abs()
        for gene, st in status.items():
            o[gene + "_mut"] = o["cell_id"].map(st).fillna(False)
            for flag, sub in o.groupby(gene + "_mut"):
                rows.append(dict(gene=gene, mutated=bool(flag), fold=f,
                                 mae=float(sub["abs_err"].mean()), n=len(sub)))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    summary = {}
    for gene in status:
        mut = df[(df.gene == gene) & df.mutated]
        wt = df[(df.gene == gene) & ~df.mutated]
        if len(mut) and len(wt):
            # per-sample test on pooled folds
            per = []
            for f in fold_list:
                try:
                    o = pd.read_parquet(OUT / f"fold{f}_predictions.parquet")
                except Exception:
                    continue
                o["abs_err"] = (o["y_true"] - o["OrgDrug-Ensemble"]).abs()
                o[gene + "_mut"] = o["cell_id"].map(status[gene]).fillna(False)
                a = o.loc[o[gene + "_mut"], "abs_err"].values
                b = o.loc[~o[gene + "_mut"], "abs_err"].values
                if len(a) > 5 and len(b) > 5:
                    stat, p = mannwhitneyu(a, b, alternative="two-sided")
                    per.append(dict(mae_mut=float(a.mean()), mae_wt=float(b.mean()),
                                    gap=float(a.mean() - b.mean()), p=float(p), fold=f))
            summary[gene] = dict(
                mean_mae_mut=float(mut.mae.mean()), mean_mae_wt=float(wt.mae.mean()),
                per_fold=per,
            )
    return summary


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"{'='*70}\n  OrgDrug LCLO  variant={args.variant}  device={args.device}\n{'='*70}", flush=True)

    data = load_gdsc_lclo(n_splits=5, seed=42)
    cell_feats, drug_feats, n_path, fp_dim, n_vocab, gene2idx = build_entity_features(data)
    path_dim = n_path

    seeds = [int(s) for s in args.seeds.split(",")]
    fold_list = [int(f) for f in args.folds.split(",")]
    fold_evals = {}

    for fold_idx in fold_list:
        tr_mask, te_mask = data.folds[fold_idx]
        print(f"\n  FOLD {fold_idx}  train={tr_mask.sum():,}  test={te_mask.sum():,}", flush=True)
        y_te = data.y[te_mask]
        seed_preds = []
        for seed in seeds:
            tt = time.time()
            print(f"    seed={seed} ...", end="", flush=True)
            init_ckpt = None
            if args.init_ckpt:
                cks = args.init_ckpt.split(",")
                init_ckpt = cks[seeds.index(seed) % len(cks)].strip()
            preds = train_one(data, cell_feats, drug_feats, fp_dim, n_vocab, path_dim,
                              tr_mask, te_mask, seed, args.device, VAR,
                              init_ckpt=init_ckpt)
            ev = compute_full_eval(y_te, preds, data.drug_names[te_mask],
                                   data.cell_ids[te_mask])
            print(f"  Pooled_r={ev['pooled_pearson']:.4f}  "
                  f"PerDrug={ev['per_drug']['mean']:.4f}  [{time.time()-tt:.0f}s]", flush=True)
            seed_preds.append(preds)

        ens = np.mean(seed_preds, axis=0).astype(np.float32)
        ev = compute_full_eval(y_te, ens, data.drug_names[te_mask],
                               data.cell_ids[te_mask])
        print(f"    ENSEMBLE  Pooled_r={ev['pooled_pearson']:.4f}  "
              f"PerDrug={ev['per_drug']['mean']:.4f}  AUROC30={ev['auroc_30']}", flush=True)

        cols = {"cell_id": data.cell_ids[te_mask],
                "drug_name": data.drug_names[te_mask], "y_true": y_te}
        for s, p in zip(seeds, seed_preds):
            cols[f"OrgDrug-s{s}"] = p
        cols["OrgDrug-Ensemble"] = ens
        pd.DataFrame(cols).to_parquet(OUT / f"fold{fold_idx}_predictions.parquet", index=False)

        fold_evals[str(fold_idx)] = dict(
            pooled_pearson=ev["pooled_pearson"], pooled_rmse=ev["pooled_rmse"],
            auroc_20=ev["auroc_20"], auroc_30=ev["auroc_30"], auroc_50=ev["auroc_50"],
            per_drug_mean=ev["per_drug"]["mean"], per_drug_median=ev["per_drug"]["median"],
            per_drug_values=list(np.asarray(ev["per_drug"]["values"], dtype=float)),
            per_cell_mean=ev["per_cell"]["mean"],
        )
        with open(OUT / "fold_evals.json", "w") as f:
            json.dump(fold_evals, f, indent=1)

    # summary
    if fold_evals:
        pm = np.mean([e["pooled_pearson"] for e in fold_evals.values()])
        dm = np.mean([e["per_drug_mean"] for e in fold_evals.values()])
        a30 = np.mean([e["auroc_30"] for e in fold_evals.values()])
        summary = dict(variant=args.variant, folds=fold_list, seeds=seeds,
                       pooled_r=float(pm), per_drug_r=float(dm), auroc30=float(a30),
                       gdrnet_reference=dict(pooled_r=0.8894, per_drug_r=0.5448, auroc30=0.9395))
        cmp_ = compare_with_gdrnet(OUT, fold_list)
        if cmp_:
            summary["paired_vs_gdrnet"] = cmp_
        mut = mutation_stratified(OUT, fold_list, gene2idx)
        if mut:
            summary["mutation_stratified"] = mut
        with open(OUT / "summary.json", "w") as f:
            json.dump(summary, f, indent=1)
        print(f"\n{'='*70}\n  SUMMARY variant={args.variant}\n"
              f"  pooled_r={pm:.4f} (GDRNet 0.8894)   per_drug_r={dm:.4f} (GDRNet 0.5448)\n"
              f"  AUROC30={a30:.4f} (GDRNet 0.9395)", flush=True)
        if cmp_:
            print(f"  paired per-drug: OrgDrug {cmp_['orgdrug_perdrug']:.4f} vs "
                  f"GDRNet {cmp_['gdrnet_perdrug']:.4f}  Δ={cmp_['delta']:+.4f}  "
                  f"p={cmp_['wilcoxon_p']:.2e}", flush=True)
        if mut:
            for g, s in mut.items():
                per = s.get("per_fold", [])
                if per:
                    gap = np.mean([x["gap"] for x in per])
                    p = min(x["p"] for x in per)
                    print(f"  {g}: MAE mut {s['mean_mae_mut']:.4f} vs wt {s['mean_mae_wt']:.4f} "
                          f"(mean gap {gap:+.4f}, best p={p:.1e})", flush=True)
    print(f"\n  total {(time.time()-t0)/60:.1f} min  ->  {OUT}", flush=True)


if __name__ == "__main__":
    main()
