# -*- coding: utf-8 -*-
"""
Merged-Ensemble Analysis — combine prediction runs into one grand ensemble
============================================================================
Merges per-fold prediction parquets from multiple runs (e.g. lclo_full_v2
5-seed + lclo_classic 3-seed = 8-model ensemble), computes metrics per fold,
compares against GDRNet v1 (results/tables/lclo_rerun) with paired Wilcoxon
across all drug x fold pairs, and reports top-k prioritization.

Usage:
  python merge_ensemble.py --runs lclo_full_v2,lclo_classic --name grand8
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(os.environ.get("ORGDRUG_ROOT",
                 Path(__file__).resolve().parents[1]))
REF = ROOT / "results/tables/lclo_rerun"
OUTROOT = ROOT / "results/orgdrug"


def per_drug_r(df, col):
    rs = {}
    for d, sub in df.groupby("drug_name"):
        if len(sub) >= 5 and sub["y_true"].std() > 1e-9 and sub[col].std() > 1e-9:
            rs[d] = float(np.corrcoef(sub["y_true"], sub[col])[0, 1])
    return rs


def auroc30(y, p):
    from sklearn.metrics import roc_auc_score
    thr = np.percentile(y, 30)
    return float(roc_auc_score((y <= thr).astype(int), -p))


def topk(df, pred_col, ks=(1, 3, 5)):
    rows = []
    for cell, sub in df.groupby("cell_id"):
        if len(sub) < ks[-1] + 1:
            continue
        sub = sub.sort_values(pred_col)
        best = int(np.argmin(sub["y_true"].rank().values))
        pr = np.argsort(np.argsort(sub[pred_col].values))[best] + 1
        rows.append({f"top{k}": int(pr <= k) for k in ks}
                    | {"spearman": float(sub[pred_col].corr(sub["y_true"], method="spearman"))})
    per = pd.DataFrame(rows)
    return {f"top{k}": round(float(per[f"top{k}"].mean()), 4) for k in ks} | {
        "spearman": round(float(per["spearman"].mean()), 4), "n_cells": len(per)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run dirs")
    ap.add_argument("--name", default="merged")
    args = ap.parse_args()

    merged_folds = {}
    for run in args.runs.split(","):
        d = OUTROOT / run
        for f in sorted(d.glob("fold*_predictions.parquet")):
            g = merged_folds.setdefault(f.name, [])
            df = pd.read_parquet(f)
            pred_cols = [c for c in df.columns if c.startswith("OrgDrug-s")]
            g.append((run, df[["cell_id", "drug_name", "y_true"] + pred_cols]))

    agg_rows, drug_r_ours, drug_r_v1 = [], {}, {}
    for fname, parts in sorted(merged_folds.items()):
        base = parts[0][1][["cell_id", "drug_name", "y_true"]].copy()
        for run, df in parts:
            for c in [c for c in df.columns if c.startswith("OrgDrug-s")]:
                base[f"{run}::{c}"] = df[c].values
        pred_cols = [c for c in base.columns if "::OrgDrug-s" in c]
        base["ens"] = base[pred_cols].mean(axis=1)

        y = base["y_true"].values
        pooled = float(np.corrcoef(y, base["ens"].values)[0, 1])
        pdm = np.mean(list(per_drug_r(base, "ens").values()))
        a30 = auroc30(y, base["ens"].values)
        agg_rows.append(dict(fold=fname, n_models=len(pred_cols), pooled=pooled,
                             per_drug=pdm, auroc30=a30))

        for d, r in per_drug_r(base, "ens").items():
            drug_r_ours.setdefault(d, []).append(r)
        ref = pd.read_parquet(REF / fname)
        m = base.merge(ref[["cell_id", "drug_name", "GDRNet-Ensemble"]],
                       on=["cell_id", "drug_name"])
        for d, r in per_drug_r(
                m.rename(columns={"y_true": "y_true", "GDRNet-Ensemble": "g1"}), "g1").items():
            drug_r_v1.setdefault(d, []).append(r)

    summary = pd.DataFrame(agg_rows)
    print(summary.to_string(index=False))
    final = dict(
        name=args.name, runs=args.runs.split(","),
        n_models=int(summary.n_models.iloc[0]) if len(summary) else 0,
        folds=len(summary),
        pooled_r=round(float(summary.pooled.mean()), 4),
        per_drug_r=round(float(summary.per_drug.mean()), 4),
        auroc30=round(float(summary.auroc30.mean()), 4),
        gdrnet=dict(pooled_r=0.8894, per_drug_r=0.5448, auroc30=0.9395),
    )

    # paired Wilcoxon on per-drug r (mean per drug)
    common = sorted(set(drug_r_ours) & set(drug_r_v1))
    if len(common) > 10:
        a = np.array([np.mean(drug_r_ours[d]) for d in common])
        b = np.array([np.mean(drug_r_v1[d]) for d in common])
        stat, p = wilcoxon(a, b)
        final["paired"] = dict(n_drugs=len(common),
                               ours=float(a.mean()), gdrnet=float(b.mean()),
                               delta=float((a - b).mean()), p=float(p),
                               n_better=int((a > b).sum()))

    # top-k on the last-fold sample (representative)
    last = sorted(merged_folds)[-1] if merged_folds else None
    if last:
        parts = merged_folds[last]
        base = parts[0][1][["cell_id", "drug_name", "y_true"]].copy()
        for run, df in parts:
            for c in [c for c in df.columns if c.startswith("OrgDrug-s")]:
                base[f"{run}::{c}"] = df[c].values
        pcs = [c for c in base.columns if "::OrgDrug-s" in c]
        base["ens"] = base[pcs].mean(axis=1)
        final["topk_fold0"] = topk(base, "ens")

    print(json.dumps(final, indent=1))
    with open(OUTROOT / f"{args.name}_summary.json", "w") as f:
        json.dump(final, f, indent=1)
    print(f"-> {OUTROOT/(args.name + '_summary.json')}")


if __name__ == "__main__":
    main()
