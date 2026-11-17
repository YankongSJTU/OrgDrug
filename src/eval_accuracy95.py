# -*- coding: utf-8 -*-
"""
95% goal gap analysis — decision-accuracy framing
=================================================
The project goal file defines the 95% target as "drug efficacy/toxicity
prediction accuracy", explicitly framing it via macro-classification
(AUROC on subsets) and top-k recommendation. This script quantifies where
we stand on those decision-accuracy metrics:

  auroc_extreme : AUROC separating true sensitive (bottom 25% lnIC50) vs
                  true resistant (top 25%) — "can the model tell the
                  extremes apart", the clinically actionable question
  auroc_extreme10 : same with bottom/top 10%
  acc@clin       : binary accuracy of sensitive/resistant calls at the
                  cohort median split (predictions median-thresholded)
  topk           : per cell-line/organoid, hit rate of the true best
                  drug being ranked in top-1/2/3

Runs on LCLO grand-11 ensemble + organoid dual-source ensembles.
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("ORGDRUG_ROOT",
                 Path(__file__).resolve().parents[1]))
OUT = ROOT / "results/orgdrug"


def auroc_extreme(y, p, q=0.25):
    lo, hi = np.quantile(y, q), np.quantile(y, 1 - q)
    m = (y <= lo) | (y >= hi)
    if m.sum() < 20:
        return float("nan")
    return float(roc_auc_score((y[m] <= lo).astype(int), -p[m]))


def acc_median(y, p):
    yp = (y <= np.median(y)).astype(int)
    pp = (p <= np.median(p)).astype(int)
    return float((yp == pp).mean())


def topk_hit(y, p, group, ks=(1, 2, 3)):
    rows = []
    for g, sub in pd.DataFrame({"y": y, "p": p, "g": group}).groupby("g"):
        if len(sub) < 6:
            continue
        order = sub["p"].values.argsort()  # ascending: predicted most sensitive first
        pos = int(np.where(order == int(np.argmin(sub["y"].values)))[0][0]) + 1
        rows.append({f"top{k}": int(pos <= k) for k in ks})
    per = pd.DataFrame(rows)
    return {f"top{k}": round(float(per[f"top{k}"].mean()), 4) for k in ks} | {
        "n_groups": len(per)}


def eval_organoid(cancer, tags=("prism5", "v1init")):
    dfs = []
    for t in tags:
        f = OUT / f"organoid_{cancer}_{t}_predictions.csv"
        if not f.exists():
            f = OUT / f"organoid_{cancer}_prism3_predictions.csv"
        if not f.exists():
            return None
        dfs.append(pd.read_csv(f))
    y = dfs[0]["y_true"].values
    ens = np.mean([d["pred_ensemble"].values for d in dfs], axis=0)
    return dict(
        n=int(len(y)),
        auroc_extreme=auroc_extreme(y, ens, 0.25),
        auroc_extreme10=auroc_extreme(y, ens, 0.10),
        acc_median=acc_median(y, ens),
        topk=topk_hit(y, ens, dfs[0]["organoid_id"].values),
    )


def eval_lclo(runs=("lclo_full_v2", "lclo_classic", "lclo_full_prisminit")):
    per_fold = []
    for fold in range(5):
        parts = []
        for r in runs:
            f = OUT / r / f"fold{fold}_predictions.parquet"
            df = pd.read_parquet(f)
            cols = [c for c in df.columns if c.startswith("OrgDrug-s")]
            parts.append(df[cols].values)
        ens = np.mean(np.concatenate(parts, axis=1), axis=1)
        per_fold.append(pd.DataFrame({
            "y": df["y_true"].values, "p": ens,
            "g": df["cell_id"].values}))
    allf = pd.concat(per_fold, ignore_index=True)
    # metrics computed per fold then averaged (honest CV estimate)
    res = {}
    for name, fn in [("auroc_extreme", lambda d: auroc_extreme(d["y"].values, d["p"].values, 0.25)),
                     ("auroc_extreme10", lambda d: auroc_extreme(d["y"].values, d["p"].values, 0.10)),
                     ("acc_median", lambda d: acc_median(d["y"].values, d["p"].values))]:
        res[name] = round(float(np.mean([fn(d) for d in per_fold])), 4)
    tk = [topk_hit(d["y"].values, d["p"].values, d["g"].values) for d in per_fold]
    res["topk"] = {k: round(float(np.mean([t[k] for t in tk])), 4)
                   for k in ("top1", "top2", "top3")}
    res["n"] = int(len(allf))
    return res


if __name__ == "__main__":
    report = {"LCLO_grand11": eval_lclo()}
    for c in ("crc", "pdac", "blca"):
        r = eval_organoid(c)
        if r:
            report[f"{c}_dual"] = r
    print(json.dumps(report, indent=1))
    with open(OUT / "accuracy95_gap.json", "w") as f:
        json.dump(report, f, indent=1)
    print("->", OUT / "accuracy95_gap.json")
