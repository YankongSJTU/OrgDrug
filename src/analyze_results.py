# -*- coding: utf-8 -*-
"""
OrgDrug Result Analysis — clinical prioritization metrics
==========================================================
Beyond correlation: Top-k hit rates (v1 Figure 8 protocol) + per-cell Spearman
+ paired comparison vs GDRNet on every metric.

Usage:
  python analyze_results.py --run lclo_full          # after E1 finishes
  python analyze_results.py --run lclo_no-mut        # ablation
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


def topk_metrics(df, pred_col="pred", k_list=(1, 3, 5)):
    """Per-cell: is the true most-sensitive drug within top-k predictions?
    Sensitivity = lowest lnIC50. Random/drug-mean baselines included."""
    rows = []
    for cell, sub in df.groupby("cell_id"):
        sub = sub.sort_values(pred_col)
        n = len(sub)
        if n < k_list[-1] + 1:
            continue
        true_rank = sub["y_true"].rank().values
        best_true = np.argmin(true_rank)
        pred_rank = np.argsort(np.argsort(sub[pred_col].values))
        pr = pred_rank[best_true] + 1
        rows.append(dict(cell=cell, n_drugs=n,
                         **{f"top{k}": int(pr <= k) for k in k_list},
                         spearman=float(pd.Series(sub[pred_col]).corr(
                             sub["y_true"], method="spearman"))))
    per = pd.DataFrame(rows)
    out = {"n_cells": len(per)}
    for k in k_list:
        out[f"top{k}_hit"] = float(per[f"top{k}"].mean())
        rand = {1: 1 / np.mean(per["n_drugs"]), 3: 3 / np.mean(per["n_drugs"]),
                5: 5 / np.mean(per["n_drugs"])}
        out[f"top{k}_random"] = float(rand[k])
    out["spearman_mean"] = float(per["spearman"].mean())
    # drug-mean baseline: rank drugs by training mean IC50 — approximate with
    # global drug mean over this fold (optimistic lower bound for baseline)
    dmean = df.groupby("drug_name")["y_true"].mean()
    df2 = df.copy()
    df2["dmean"] = df2["drug_name"].map(dmean)
    rows = []
    for cell, sub in df2.groupby("cell_id"):
        if len(sub) < 6:
            continue
        sub = sub.sort_values("dmean")
        best_true = np.argmin(sub["y_true"].rank().values)
        pr = np.argsort(np.argsort(sub["dmean"].values))[best_true] + 1
        rows.append(dict(cell=cell, **{f"top{k}": int(pr <= k) for k in k_list}))
    dm = pd.DataFrame(rows)
    for k in k_list:
        out[f"top{k}_drugmean"] = float(dm[f"top{k}"].mean())
    return out


def per_drug_r(df, pred_col):
    rs = {}
    for d, sub in df.groupby("drug_name"):
        if len(sub) >= 5 and sub["y_true"].std() > 1e-9 and sub[pred_col].std() > 1e-9:
            rs[d] = float(np.corrcoef(sub["y_true"], sub[pred_col])[0, 1])
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="lclo_full")
    args = ap.parse_args()
    run_dir = ROOT / f"results/orgdrug/{args.run}"

    folds = sorted(run_dir.glob("fold*_predictions.parquet"))
    print(f"run={args.run}  folds found={len(folds)}")
    if not folds:
        return

    ours_rows, ref_rows = [], []
    for f in folds:
        o = pd.read_parquet(f)
        ours_rows.append(o)
        if (REF / f.name).exists():
            g = pd.read_parquet(REF / f.name)
            ref_rows.append(g)

    ours = pd.concat(ours_rows)
    ens = ours[["cell_id", "drug_name", "y_true", "OrgDrug-Ensemble"]].rename(
        columns={"OrgDrug-Ensemble": "pred"})

    result = dict(run=args.run, n_pairs=len(ens))
    result["prioritization"] = topk_metrics(ens)

    if ref_rows:
        ref = pd.concat(ref_rows)[["cell_id", "drug_name", "y_true",
                                   "GDRNet-Ensemble"]].rename(
            columns={"GDRNet-Ensemble": "pred"})
        result["gdrnet_prioritization"] = topk_metrics(ref)

        # paired per-drug r across all folds
        ro = {}
        for (fi, o) in zip(folds, ours_rows):
            for d, r in per_drug_r(o.rename(columns={"OrgDrug-Ensemble": "pred"}),
                                   "pred").items():
                ro.setdefault(d, []).append(r)
        rg = {}
        for (fi, g) in zip(folds, ref_rows):
            for d, r in per_drug_r(g.rename(columns={"GDRNet-Ensemble": "pred"}),
                                   "pred").items():
                rg.setdefault(d, []).append(r)
        common = sorted(set(ro) & set(rg))
        a = np.array([np.mean(ro[d]) for d in common])
        b = np.array([np.mean(rg[d]) for d in common])
        stat, p = wilcoxon(a, b)
        result["paired_per_drug"] = dict(
            n_drugs=len(common), orgdrug=float(a.mean()), gdrnet=float(b.mean()),
            delta=float((a - b).mean()), p=float(p),
            n_better=int((a > b).sum()), n_worse=int((a < b).sum()))

    print(json.dumps(result, indent=1))
    with open(run_dir / "prioritization.json", "w") as f:
        json.dump(result, f, indent=1)
    print(f"-> {run_dir/'prioritization.json'}")


if __name__ == "__main__":
    main()
