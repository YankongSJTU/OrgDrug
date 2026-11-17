# -*- coding: utf-8 -*-
"""Rebuild driver mutation matrix using full MAF impact flags.

The previous matrix (OmicsSomaticMutations_damaging subset) missed canonical
oncogene activations (KRAS G12D etc. are missense, not 'damaging'). v2 uses:
    OncogeneHighImpact | TumorSuppressorHighImpact | LikelyLoF
which captures both activating hotspots and loss-of-function.
Overwrites data/processed/orgdrug/mut_matrix.npy + mut_cell_ids.npy.
"""
import numpy as np
import pandas as pd
import os
from pathlib import Path

ROOT = Path(os.environ.get("ORGDRUG_ROOT",
                 Path(__file__).resolve().parents[1]))
FEAT = ROOT / "data/processed/orgdrug"

maf = pd.read_csv(
    ROOT / "data/raw/ccle/OmicsSomaticMutations.csv",
    usecols=["ModelID", "HugoSymbol", "ProteinChange", "VariantType",
             "OncogeneHighImpact", "TumorSuppressorHighImpact", "LikelyLoF"],
    low_memory=False)
print("maf rows:", len(maf), flush=True)
for c in ["OncogeneHighImpact", "TumorSuppressorHighImpact", "LikelyLoF"]:
    print(c, maf[c].astype(str).value_counts().head(3).to_dict(), flush=True)


def _true(s):
    return s.astype(str).str.lower().eq("true")


keep = (_true(maf["OncogeneHighImpact"]) | _true(maf["TumorSuppressorHighImpact"])
        | _true(maf["LikelyLoF"]))

# DepMap's impact flags are conservative (KRAS G12D = False on all three).
# Add recurrent-hotspot rule: driver-gene protein changes seen in >=3 cell
# lines (G12D ~60 lines; random passengers don't recur).
drivers_set = set(pd.read_csv(FEAT / "mut_genes.csv")["gene"])
maf["gene"] = maf["HugoSymbol"]
rec = (maf[maf["gene"].isin(drivers_set) & maf["ProteinChange"].notna()]
       .groupby(["gene", "ProteinChange"])["ModelID"].nunique())
hotspots = set(rec[rec >= 3].index)
is_hot = pd.Series(False, index=maf.index)
if hotspots:
    keys = list(zip(maf["gene"], maf["ProteinChange"].fillna("")))
    is_hot = pd.Series(keys, index=maf.index).isin(hotspots)
keep = keep | is_hot

sub = maf[keep][["ModelID", "HugoSymbol"]].drop_duplicates()
print("impact+hotspot variants:", len(sub), "genes:", sub.HugoSymbol.nunique(),
      "cells:", sub.ModelID.nunique(), flush=True)

genes_df = pd.read_csv(FEAT / "mut_genes.csv")
drivers = genes_df["gene"].tolist()
sub = sub[sub["HugoSymbol"].isin(drivers)]
mat = pd.crosstab(sub["ModelID"], sub["HugoSymbol"]).reindex(
    columns=drivers, fill_value=0)
out = pd.DataFrame(0, index=mat.index, columns=drivers, dtype=np.float32)
out.loc[mat.index, drivers] = np.minimum(mat[drivers].values, 1)

for g in ["KRAS", "TP53", "BRAF", "EGFR", "PIK3CA"]:
    if g in out:
        print(f"  {g}: {int(out[g].sum())}/{len(out)} cells", flush=True)
np.save(FEAT / "mut_matrix.npy", out.values.astype(np.float32))
np.save(FEAT / "mut_cell_ids.npy", out.index.astype(str).values)
print("saved mut_matrix.npy (v2 impact flags)", flush=True)
