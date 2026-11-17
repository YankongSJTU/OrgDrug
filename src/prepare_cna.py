# -*- coding: utf-8 -*-
"""
E5c — CNA (copy-number) view preparation
=========================================
Builds a copy-number matrix over the SAME driver-gene panel as the mutation
view, from the DepMap 24Q2 OmicsCNGene.csv (log2 absolute CN, neutral ~= 1.0).

Transform: delta = value - 1.0  (log2 gain/loss relative to diploid),
clipped to [-2, 2]. Missing gene/cell -> 0 (no change) and per-cell avail
flag stored separately.

Outputs -> data/processed/orgdrug/:
  cna_matrix.npy     (cells x n_driver_genes, float32)
  cna_cell_ids.npy   (ModelID per row)
  cna_genes.csv      (gene list == mut_genes.csv ordering, for alignment)
"""

import os
import io
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ORGDRUG_ROOT",
                 Path(__file__).resolve().parents[1]))
HF = ROOT / "data/raw/ccle/hf"
FEAT = ROOT / "data/processed/orgdrug"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass


def log(m):
    print(f"[cna-prep {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    if (FEAT / "cna_matrix.npy").exists():
        log("cna matrix exists, skipping (delete to rebuild)")
        return

    genes = pd.read_csv(FEAT / "mut_genes.csv")["gene"].tolist()
    log(f"driver gene panel: {len(genes)} (aligned with mut view)")

    log("loading OmicsCNGene.csv (~800MB, a few minutes) ...")
    df = pd.read_csv(HF / "OmicsCNGene.csv", index_col=0, low_memory=False)
    sym = {c: c.split(" (")[0].strip() for c in df.columns
           if re.match(r"^[A-Za-z0-9._-]+ \(\d+\)$", c)}
    df = df[list(sym)].rename(columns=sym)
    if "ModelID" in df.columns:
        pass
    df.index = df.index.astype(str)
    log(f"CNA matrix: {df.shape} cells x genes")

    have = [g for g in genes if g in df.columns]
    log(f"panel genes present: {len(have)}/{len(genes)}")
    sub = df[have].astype(np.float32)
    # log2 absolute CN -> delta vs diploid(=1.0), clipped; NaN -> 0 (neutral)
    delta = np.clip(sub.values - 1.0, -2.0, 2.0)
    delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    # cells with all-zero rows (missing CNA) flagged via row max
    ok = np.abs(delta).max(axis=1) > 0
    log(f"cells with non-trivial CNA profile: {ok.sum()}/{len(delta)}")

    np.save(FEAT / "cna_matrix.npy", delta.astype(np.float32))
    np.save(FEAT / "cna_cell_ids.npy", df.index.values.astype(str))
    pd.DataFrame({"gene": genes,
                  "present": [g in have for g in genes]}).to_csv(
        FEAT / "cna_genes.csv", index=False)
    amp = (delta > 0.3).sum(axis=0).mean()
    dele = (delta < -0.3).sum(axis=0).mean()
    log(f"mean amp genes/cell(>0.3): {amp:.0f}  del genes/cell(<-0.3): {dele:.0f}")
    log(f"saved -> cna_matrix.npy {delta.shape}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
