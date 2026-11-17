# -*- coding: utf-8 -*-
"""
OrgDrug Pretraining on the Expanded GDSC Set (964 cell lines)
==============================================================
Trains OrgDrug on the same expanded pair set / split masks used by GDRNet v1
(data/processed/expanded/*), saving per-seed checkpoints for organoid transfer
(E4) and PRISM-combined pretraining (E5).

Usage:
  python pretrain_orgdrug_expanded.py --device cuda:6 --seeds 42,123,456
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
EXP = ROOT / "data/processed/expanded"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass

from models.orgdrug import OrgDrug, OrgDrugDataset, orgdrug_loss  # noqa: E402
from orgdrug_data import (load_tables, cell_set_rows, drug_set_rows,  # noqa: E402
                          drug_train_stats, compute_pair_weights)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:6")
    p.add_argument("--data-dir", default=str(EXP),
                   help="dataset dir in expanded/ format; expanded_prism for E5")
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n-cross", type=int, default=3)
    p.add_argument("--variant", default="full", choices=["full", "classic"])
    p.add_argument("--lambda-list", type=float, default=0.05)
    p.add_argument("--lambda-cls", type=float, default=0.2)
    p.add_argument("--view-dropout", type=float, default=0.05)
    p.add_argument("--organoid-augment", type=float, default=0.0,
                   help="p(mask gene+mut+path jointly). WARNING: teaching the "
                        "model to emit drug-means for masked samples — use 0 "
                        "for transfer-leg pretraining (v1 recipe)")
    p.add_argument("--dump-val", action="store_true",
                   help="dump best-model val predictions to "
                        "results/orgdrug/tox_pretrain/<prefix>_s{seed}.parquet")
    return p.parse_args()


def main():
    args = parse_args()
    DATA = Path(args.data_dir)
    suffix = "expanded" if DATA.name == "expanded" else DATA.name.replace("expanded_", "")
    print("=" * 70 + f"\n  OrgDrug pretraining  ({DATA.name})\n" + "=" * 70, flush=True)
    t0 = time.time()

    X_gene = np.load(DATA / "X_gene.npy").astype(np.float32)
    X_scf = np.load(DATA / "X_scf.npy").astype(np.float32)
    X_fp = np.load(DATA / "X_fp.npy").astype(np.float32)
    cell_idx = np.load(DATA / "cell_idx.npy").astype(np.int64)
    drug_idx = np.load(DATA / "drug_idx.npy").astype(np.int64)
    y = np.load(DATA / "y.npy").astype(np.float32)
    tr_mask = np.load(DATA / "tr_mask.npy").astype(bool)
    val_mask = np.load(DATA / "val_mask.npy").astype(bool)
    meta = pd.read_csv(DATA / "expanded_metadata.csv")
    meta = meta.rename(columns={"DRUG_NAME": "drug_name"})
    assert len(meta) == len(y), f"meta {len(meta)} != y {len(y)}"

    cell_map = pd.read_csv(DATA / "cell_id_map.csv")
    drug_map = pd.read_csv(DATA / "drug_id_map.csv")

    # entity lists in embedding-index order (1-based -> position i-1)
    cell_ids_by_idx = [""] * (cell_map.shape[0] + 1)
    for _, r in cell_map.iterrows():
        cell_ids_by_idx[int(r[cell_map.columns[1]])] = str(r[cell_map.columns[0]])
    drug_names_by_idx = [""] * (drug_map.shape[0] + 1)
    for _, r in drug_map.iterrows():
        drug_names_by_idx[int(r[drug_map.columns[1]])] = str(r[drug_map.columns[0]])
    # BUG FIX (2026-11-05): legacy cell_id_map.csv is keyed by COSMIC_ID while
    # feature tables are keyed by ModelID -> every cell resolved to pair row 0
    # and mut/path were all-zero. Translate COSMIC -> ModelID; ModelID-keyed
    # maps (expanded_prism) pass through unchanged.
    if "COSMIC_ID" in cell_map.columns and "ModelID" in meta.columns:
        c2m = meta.drop_duplicates("COSMIC_ID").set_index("COSMIC_ID")["ModelID"].astype(str)
        cell_list = [c2m.get(c, c) for c in cell_ids_by_idx[1:]]
    else:
        cell_list = cell_ids_by_idx[1:]
    drug_list = drug_names_by_idx[1:]
    scf_avail_pairs = None
    if (DATA / "scf_avail.npy").exists():
        scf_avail_pairs = np.load(DATA / "scf_avail.npy").astype(np.float32)
    print(f"  {DATA.name}: {len(y):,} pairs, {cell_map.shape[0]} cells, "
          f"{drug_map.shape[0]} drugs, train={tr_mask.sum():,} val={val_mask.sum():,}"
          f"{'' if scf_avail_pairs is None else f', scf_avail={(scf_avail_pairs>0).mean()*100:.0f}%'}",
          flush=True)

    t = load_tables()
    path_by_cell, mut_idx, mut_mask = cell_set_rows(t, cell_list)

    # per-pair matrices -> per-entity first-occurrence rows
    def _first_rows(keys, target):
        first = pd.Series(np.arange(len(meta)), index=keys).groupby(level=0).first()
        return first.reindex(target).fillna(0).astype(np.int64).values

    gene_by_cell = X_gene[_first_rows(meta["ModelID"].values, cell_list)]
    scf_by_cell = X_scf[_first_rows(meta["ModelID"].values, cell_list)]

    first_drug = pd.Series(np.arange(len(meta)), index=meta["drug_name"].values).groupby(level=0).first()
    morgan_by_drug = {d: X_fp[int(first_drug.reindex([d]).fillna(0).iloc[0])]
                      for d in drug_list}
    fp_by_drug, tgt_idx, tgt_mask = drug_set_rows(t, drug_list, morgan_by_drug)

    cell_feats = dict(gene=gene_by_cell, scf=scf_by_cell, path=path_by_cell,
                      mut_idx=mut_idx, mut_mask=mut_mask)
    drug_feats = dict(fp=fp_by_drug, tgt_idx=tgt_idx, tgt_mask=tgt_mask)

    drug_has_tgt = drug_feats["tgt_mask"].any(axis=1)
    avail = np.ones((len(y), 8), dtype=np.float32)
    avail[:, 5] = drug_has_tgt[drug_idx - 1]
    if scf_avail_pairs is not None:
        avail[:, 1] = scf_avail_pairs

    stats, gm, gs = drug_train_stats(y, meta["drug_name"].values, tr_mask)
    w, sens = compute_pair_weights(y, meta["drug_name"].values, stats, gm, gs)

    tr_ds = OrgDrugDataset(cell_feats, drug_feats, cell_idx[tr_mask] - 1,
                           drug_idx[tr_mask] - 1, y[tr_mask],
                           w[tr_mask], sens[tr_mask], avail[tr_mask],
                           emb_cell_idx=cell_idx[tr_mask],
                           emb_drug_idx=drug_idx[tr_mask])
    va_ds = OrgDrugDataset(cell_feats, drug_feats, cell_idx[val_mask] - 1,
                           drug_idx[val_mask] - 1, y[val_mask],
                           w[val_mask], sens[val_mask], avail[val_mask],
                           emb_cell_idx=cell_idx[val_mask],
                           emb_drug_idx=drug_idx[val_mask])

    from torch.utils.data import DataLoader
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                           num_workers=args.workers, pin_memory=True,
                           persistent_workers=args.workers > 0)
    va_loader = DataLoader(va_ds, batch_size=args.batch * 2, shuffle=False,
                           num_workers=2, pin_memory=True,
                           persistent_workers=args.workers > 0)

    device = args.device
    for seed in [int(s) for s in args.seeds.split(",")]:
        ts = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        if args.variant == "classic":
            from models.orgdrug_classic import OrgDrugClassic

            class _WrappedClassic(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, *b):
                    return self.m(*b), None

            model = _WrappedClassic(OrgDrugClassic(
                fp_dim=t["fp_dim"], path_dim=t["path_dim"],
                n_shared_genes=t["n_vocab"],
                n_cells=cell_map.shape[0], n_drugs=drug_map.shape[0],
                n_cross=args.n_cross, dropout=args.dropout).to(device))
        else:
            model = OrgDrug(
                n_cells=cell_map.shape[0], n_drugs=drug_map.shape[0],
                fp_dim=t["fp_dim"], path_dim=t["path_dim"],
                n_shared_genes=t["n_vocab"], n_cross=args.n_cross,
                dropout=args.dropout).to(device)
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

        best_rmse, best_state, wait = float("inf"), None, 0
        for epoch in range(args.epochs):
            model.train()
            for (xg, scf, fp, path, mi, mm, ti, tm, ci, di, av, yb, wb, sb) in tr_loader:
                batch = [tt.to(device, non_blocking=True) for tt in
                         (xg, scf, fp, path, mi, mm, ti, tm, ci, di)]
                av, yb, wb, sb = (av.to(device), yb.to(device), wb.to(device), sb.to(device))
                if args.view_dropout > 0:  # random view dropout
                    drop = torch.rand(av.shape, device=device) < args.view_dropout
                    drop[:, 6:] = False
                    drop = drop & (av > 0.5)
                    av = av * (~drop).float()
                # organoid-mode augmentation (optional): simulate the transfer
                # regime — CAUTION: this teaches drug-mean behavior for masked
                # samples; keep 0 for the transfer leg (see args help)
                if args.organoid_augment > 0:
                    org_mode = torch.rand(av.shape[0], device=device) < args.organoid_augment
                    if org_mode.any():
                        av = av.clone()
                        av[org_mode, 0] = 0.0
                        av[org_mode, 2] = 0.0
                        av[org_mode, 3] = 0.0
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
                    reg, cls = model(*batch, av)
                    loss = orgdrug_loss(reg, cls, yb, wb, sb, batch[8],
                                        lambda_list=args.lambda_list,
                                        lambda_cls=args.lambda_cls,
                                        use_cls=args.lambda_cls > 0 and args.variant != "classic")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            model.eval()
            preds = []
            with torch.no_grad():
                for (xg, scf, fp, path, mi, mm, ti, tm, ci, di, av, yb, wb, sb) in va_loader:
                    batch = [tt.to(device, non_blocking=True) for tt in
                             (xg, scf, fp, path, mi, mm, ti, tm, ci, di)]
                    reg, _ = model(*batch, av.to(device))
                    preds.append(reg.float().cpu())
            preds = torch.cat(preds).numpy()
            rmse = float(np.sqrt(np.mean((y[val_mask] - preds) ** 2)))
            r = float(np.corrcoef(y[val_mask], preds)[0, 1])
            if rmse < best_rmse:
                best_rmse, wait = rmse, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= args.patience:
                    break
            if (epoch + 1) % 20 == 0:
                print(f"  s{seed} ep{epoch+1}  val_rmse={rmse:.4f} val_r={r:.4f} "
                      f"best={best_rmse:.4f}  [{time.time()-ts:.0f}s]", flush=True)

        model.load_state_dict(best_state)
        prefix = ("orgdrug_classic_" if args.variant == "classic" else "orgdrug_") + suffix
        ckpt = MODELS / f"{prefix}_s{seed}.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"  seed {seed}: best val_rmse={best_rmse:.4f}  -> {ckpt.name}  "
              f"({(time.time()-ts)/60:.1f} min)", flush=True)

        if args.dump_val:
            model.eval()
            preds = []
            with torch.no_grad():
                for (xg, scf, fp, path, mi, mm, ti, tm, ci, di, av, yb, wb, sb) in va_loader:
                    batch = [tt.to(device, non_blocking=True) for tt in
                             (xg, scf, fp, path, mi, mm, ti, tm, ci, di)]
                    reg, _ = model(*batch, av.to(device))
                    preds.append(reg.float().cpu())
            preds = torch.cat(preds).numpy()
            idx = np.where(val_mask)[0]
            dump_dir = ROOT / "results/orgdrug/tox_pretrain"
            dump_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"drug_name": meta["drug_name"].values[idx],
                          "ModelID": meta["ModelID"].values[idx],
                          "y": y[idx], "pred": preds}).to_parquet(
                dump_dir / f"{prefix}_s{seed}.parquet", index=False)
            print(f"    val preds dumped ({len(idx):,} pairs) -> {prefix}_s{seed}.parquet",
                  flush=True)

    config = dict(n_cells=cell_map.shape[0], n_drugs=drug_map.shape[0],
                  fp_dim=t["fp_dim"], path_dim=t["path_dim"],
                  n_shared_genes=t["n_vocab"], n_cross=args.n_cross,
                  variant=args.variant, dropout=args.dropout)
    cfg_name = ("orgdrug_classic_config" if args.variant == "classic"
                else "orgdrug_config") + \
        ("" if suffix == "expanded" else "_" + suffix) + ".json"
    (MODELS / cfg_name).write_text(json.dumps(config, indent=1))
    print(f"  config saved ({cfg_name}). total {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
