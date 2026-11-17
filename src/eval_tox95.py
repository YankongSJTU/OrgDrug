# -*- coding: utf-8 -*-
"""
Toxicity-endpoint 95% evaluation
================================
Reads the dumped val predictions (drug-disjoint!) from tox pretrains
(results/orgdrug/tox_pretrain/*.parquet) and reports:

  per-pair     : Pearson r, RMSE
  tox_disc     : AUROC separating true-toxic (bottom 25% AUC) vs true-benign
                 (top 25%) pairs — the "毒性判断准确率" framing (also 10%)
  per-drug     : Spearman of predicted vs actual pan-panel median AUC over
                 unseen drugs (structure-only toxicity ranking)

Usage:
  python eval_tox95.py                # both variants, 3-seed ensembles
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("ORGDRUG_ROOT",
                 Path(__file__).resolve().parents[1]))
DUMP = ROOT / "results/orgdrug/tox_pretrain"
OUT = ROOT / "results/orgdrug/tox95_summary.json"
VARIANTS = {"full": "orgdrug_prism_tox_s", "classic": "orgdrug_classic_prism_tox_s"}
SEEDS = (42, 123, 456)


def disc_auroc(y, p, q):
    lo, hi = np.quantile(y, q), np.quantile(y, 1 - q)
    m = (y <= lo) | (y >= hi)
    return float(roc_auc_score((y[m] <= lo).astype(int), -p[m]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="prism_tox",
                    help="dataset suffix (data-dir name); dumps are named "
                         "orgdrug[_classic]_<suffix>_s{seed}.parquet")
    args = ap.parse_args()
    variants = {"full": f"orgdrug_{args.suffix}_s",
                "classic": f"orgdrug_classic_{args.suffix}_s"}
    report = {}
    for vname, prefix in variants.items():
        frames = []
        for s in SEEDS:
            f = DUMP / f"{prefix}{s}.parquet"
            if f.exists():
                frames.append(pd.read_parquet(f))
        if not frames:
            print(f"[{vname}] no dumps, skip")
            continue
        base = frames[0][["drug_name", "ModelID", "y"]].copy()
        pred = np.mean([fr["pred"].values for fr in frames], axis=0)
        base["pred"] = pred
        y = base["y"].values

        per_drug = base.groupby("drug_name").agg(
            y_med=("y", "median"), p_med=("pred", "median")).reset_index()
        rho, pv = spearmanr(per_drug["y_med"], per_drug["p_med"])

        report[vname] = dict(
            n_seeds=len(frames), n_pairs=int(len(base)),
            n_unseen_drugs=int(len(per_drug)),
            pearson=round(float(np.corrcoef(y, pred)[0, 1]), 4),
            rmse=round(float(np.sqrt(np.mean((y - pred) ** 2))), 4),
            tox_disc_25=round(disc_auroc(y, pred, 0.25), 4),
            tox_disc_10=round(disc_auroc(y, pred, 0.10), 4),
            per_drug_spearman=round(float(rho), 4),
            per_drug_p=float(pv),
        )
        print(f"[{vname}] pairs={len(base):,} unseen_drugs={len(per_drug)}  "
              f"r={report[vname]['pearson']}  "
              f"disc25={report[vname]['tox_disc_25']}  "
              f"disc10={report[vname]['tox_disc_10']}  "
              f"drug_rho={report[vname]['per_drug_spearman']}")

    with open(OUT.with_name(f"tox95_{args.suffix}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("->", OUT.with_name(f"tox95_{args.suffix}.json"))


if __name__ == "__main__":
    main()
