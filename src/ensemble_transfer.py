# -*- coding: utf-8 -*-
"""
Dual-source organoid-transfer ensemble
=======================================
Averages prediction CSVs from the two transfer recipes and reports metrics
per cancer cohort:

  prism5  : OrgDrugClassic warm-started from PRISM-pretrained checkpoints
            (5 seeds), finetuned per organoid LOOCV
  v1init  : OrgDrugClassic shell warm-started from GDRNet v1 expanded
            checkpoints (3 seeds), finetuned per organoid LOOCV
  dual    : mean of the two recipe ensembles (equal weight)
  mega    : mean over all per-seed prediction columns of both recipes

Usage:
  python ensemble_transfer.py                      # prism5 + v1init
  python ensemble_transfer.py --a prism3 --b v1init
"""
import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("ORGDRUG_ROOT",
                 Path(__file__).resolve().parents[1]))
OUT = ROOT / "results/orgdrug"
V1 = {"crc": dict(r=0.8902, auroc=0.9411),
      "pdac": dict(r=0.741, auroc=0.831),
      "blca": dict(r=0.640, auroc=0.868)}


def metrics(y, p):
    thr = np.percentile(y, 30)
    try:
        auc = float(roc_auc_score((y <= thr).astype(int), -p))
    except Exception:
        auc = float("nan")
    return dict(pearson=round(float(np.corrcoef(y, p)[0, 1]), 4),
                r2=round(float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)), 4),
                rmse=round(float(np.sqrt(np.mean((y - p) ** 2))), 4),
                auroc=round(auc, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="prism5", help="tag of recipe A")
    ap.add_argument("--b", default="v1init", help="tag of recipe B")
    ap.add_argument("--cancers", default="crc,pdac,blca")
    ap.add_argument("--out", default="dual_source_summary.json")
    args = ap.parse_args()

    report = {}
    for c in args.cancers.split(","):
        fa = OUT / f"organoid_{c}_{args.a}_predictions.csv"
        fb = OUT / f"organoid_{c}_{args.b}_predictions.csv"
        if not fa.exists() or not fb.exists():
            print(f"[{c}] missing {'A' if not fa.exists() else 'B'} predictions, skip")
            continue
        A = pd.read_csv(fa)
        B = pd.read_csv(fb)
        key = ["organoid_id", "drug_name"]
        assert (A[key].values == B[key].values).all(), f"{c}: row mismatch"
        assert np.allclose(A["y_true"], B["y_true"]), f"{c}: y mismatch"
        y = A["y_true"].values

        sa = [x for x in A.columns if x.startswith("pred_s")]
        sb = [x for x in B.columns if x.startswith("pred_s")]
        dual = (A["pred_ensemble"].values + B["pred_ensemble"].values) / 2
        mega = np.mean([A[x].values for x in sa] + [B[x].values for x in sb], axis=0)

        ka, kb = f"{args.a}({len(sa)}s)", f"{args.b}({len(sb)}s)"
        km = f"mega({len(sa)}+{len(sb)}s)"
        report[c] = {
            "n_pairs": int(len(y)),
            ka: metrics(y, A["pred_ensemble"].values),
            kb: metrics(y, B["pred_ensemble"].values),
            "dual": metrics(y, dual),
            km: metrics(y, mega),
            "v1_ref": V1.get(c),
        }
        print(f"[{c}] n={len(y)}  "
              f"{args.a}={report[c][ka]['pearson']}/{report[c][ka]['auroc']}  "
              f"{args.b}={report[c][kb]['pearson']}/{report[c][kb]['auroc']}  "
              f"dual={report[c]['dual']['pearson']}/{report[c]['dual']['auroc']}  "
              f"mega={report[c][km]['pearson']}/{report[c][km]['auroc']}  "
              f"(v1 r={V1[c]['r']}, auroc={V1[c]['auroc']})")

    with open(OUT / args.out, "w") as f:
        json.dump(dict(a=args.a, b=args.b, results=report), f, indent=1)
    print(f"-> {OUT / args.out}")


if __name__ == "__main__":
    main()
