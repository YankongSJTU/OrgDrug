# -*- coding: utf-8 -*-
"""
Build data/processed/prism_tox/ — PRISM toxicity-endpoint dataset
=================================================================
Same feature matrices / entity maps as expanded_prism (drug-encoder
chemical diversity preserved), but:

  y      = PRISM secondary-screen AUC  (integrated viability loss over the
           full dose range; LOWER = more toxic) — the toxicity endpoint
  pairs  = PRISM-only, passed_str_profiling & curve r2 >= 0.3, median over
           replicate screens per (drug, cell)
  split  = --split drug (default): DRUG-DISJOINT 20% val — cold-start
           toxicity of unseen compounds from structure alone
           --split cell: CELL-DISJOINT 20% val — deployment framing
           (known drugs, unseen cell contexts) matching the organoid story

Sidecar y_lower.npy = median lower_limit (max-kill plateau, selectivity
analysis). Everything else (X_*, *_id_map) mirrors expanded_prism.

Usage:
  python prepare_prism_tox.py                     # drug split -> prism_tox/
  python prepare_prism_tox.py --split cell        # -> prism_tox_cellsplit/
"""
import os
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ORGDRUG_ROOT",
                 Path(__file__).resolve().parents[1]))
SEC = ROOT / "data/raw/ccle/hf/secondary_screen_params.csv"
SRC = ROOT / "data/processed/expanded_prism"
R2_MIN = 0.3
VAL_FRAC = 0.20


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["drug", "cell"], default="drug")
    args = ap.parse_args()
    OUT = ROOT / ("data/processed/prism_tox" if args.split == "drug"
                  else "data/processed/prism_tox_cellsplit")
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    log("reading secondary_screen_params ...")
    ss = pd.read_csv(SEC, low_memory=False)
    n0 = len(ss)
    keep = (ss["passed_str_profiling"] == True) & ss["auc"].notna() \
        & (ss["r2"] >= R2_MIN)
    log(f"  {n0:,} rows -> {keep.sum():,} after "
        f"passed_str_profiling & auc & r2>={R2_MIN} "
        f"(r2 quartiles {np.nanpercentile(ss['r2'], [25, 50, 75]).round(2)})")
    ss = ss[keep]

    log("median over replicate screens ...")
    g = ss.groupby(["broad_id", "depmap_id"]).agg(
        auc=("auc", "median"), lower=("lower_limit", "median"),
        name=("ccle_name", "first"), drug_name=("name", "first"))
    g = g.reset_index()
    log(f"  pairs={len(g):,}  drugs={g['broad_id'].nunique()}  "
        f"cells={g['depmap_id'].nunique()}")

    # ── reuse expanded_prism entity maps + features ──
    cm = pd.read_csv(SRC / "cell_id_map.csv")
    dm = pd.read_csv(SRC / "drug_id_map.csv")
    cell2idx = dict(zip(cm["ModelID"].astype(str), cm["cell_idx"]))
    drug2idx = dict(zip(dm["DRUG_NAME"].astype(str), dm["drug_idx"]))
    g["cell_row"] = g["depmap_id"].astype(str).map(cell2idx)
    g["drug_row"] = ("PR::" + g["broad_id"]).map(drug2idx)
    miss_c = g["cell_row"].isna().sum()
    miss_d = g["drug_row"].isna().sum()
    log(f"  unmapped: cells={miss_c}  drugs={miss_d} (dropped)")
    g = g.dropna(subset=["cell_row", "drug_row"])
    g["cell_row"] = g["cell_row"].astype(np.int64)
    g["drug_row"] = g["drug_row"].astype(np.int64)

    # per-cell scF availability, derived from expanded_prism pair arrays
    eci = np.load(SRC / "cell_idx.npy").astype(np.int64)
    escf = np.load(SRC / "scf_avail.npy").astype(np.float32)
    cell_has_scf = pd.Series(escf).groupby(eci).max().to_dict()

    ci = g["cell_row"].values
    di = g["drug_row"].values
    y = g["auc"].values.astype(np.float32)
    scf = np.array([cell_has_scf.get(c, 0.0) for c in ci], dtype=np.float32)

    # ── split ──
    rng = np.random.RandomState(42)
    if args.split == "drug":
        keys = np.array(sorted(g["broad_id"].unique()))
        val_keys = set(keys[rng.permutation(len(keys))[:int(len(keys) * VAL_FRAC)]])
        val_mask = g["broad_id"].isin(val_keys).values
        log(f"  drug-split: {len(val_keys)} val drugs, "
            f"val pairs={val_mask.sum():,} / train={(~val_mask).sum():,}")
    else:
        keys = np.array(sorted(g["depmap_id"].unique()))
        val_keys = set(keys[rng.permutation(len(keys))[:int(len(keys) * VAL_FRAC)]])
        val_mask = g["depmap_id"].isin(val_keys).values
        log(f"  cell-split: {len(val_keys)} val cells, "
            f"val pairs={val_mask.sum():,} / train={(~val_mask).sum():,}")

    log("linking feature matrices ...")
    for f in ("X_gene.npy", "X_scf.npy", "X_fp.npy"):
        dst = OUT / f
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(SRC.resolve() / f)
    cm.to_csv(OUT / "cell_id_map.csv", index=False)
    dm.to_csv(OUT / "drug_id_map.csv", index=False)

    np.save(OUT / "cell_idx.npy", ci)
    np.save(OUT / "drug_idx.npy", di)
    np.save(OUT / "y.npy", y)
    np.save(OUT / "y_lower.npy", g["lower"].values.astype(np.float32))
    np.save(OUT / "tr_mask.npy", ~val_mask)
    np.save(OUT / "val_mask.npy", val_mask)
    np.save(OUT / "scf_avail.npy", scf)

    pd.DataFrame({
        "COSMIC_ID": "PRISM", "CELL_LINE_NAME": g["name"].values,
        "DRUG_NAME": "PR::" + g["broad_id"].values, "LN_IC50": y,
        "ModelID": g["depmap_id"].values, "is_train": ~val_mask,
        "has_scf": scf > 0, "source": "prism_tox",
        "lower_limit": g["lower"].values, "drug_label": g["drug_name"].values,
    }).to_csv(OUT / "expanded_metadata.csv", index=False)

    q = np.percentile(y, [10, 25, 50, 75, 90]).round(3)
    log(f"auc quartiles 10/25/50/75/90: {q}  (mean {y.mean():.3f}±{y.std():.3f})")
    lo = np.nanpercentile(g["lower"].values, [25, 50, 75]).round(3)
    log(f"lower_limit quartiles 25/50/75: {lo}")
    log(f"ALL DONE {(time.time()-t0)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
