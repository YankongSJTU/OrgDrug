# -*- coding: utf-8 -*-
"""Synthetic demo dataset for OrgDrug (schema-exact, seeded, ~3 MB).

Generates every processed-level table the LCLO training leg expects:

  data/external/gdsc_drug_smiles.csv        30 valid demo structures
  data/processed/gdsc_metadata.parquet      2,400 pairs (80 cells x 30 drugs)
  data/processed/gdsc_response_lnic50.parquet
  data/processed/gdsc_cell_features.parquet 400 landmark genes per pair
  data/processed/scfoundation_cell_{emb,ids}.npy     512-d embeddings
  data/processed/orgdrug/...                mutation / pathway / MACCS /
                                            target / one-hot / shared vocab

The response carries a learnable signal: a per-drug bias plus a cell-latent x
drug-latent interaction that lives in the gene view (cell side) and the
fingerprint views (drug side), so even a 15-epoch smoke run produces sensible
metrics. KRAS mutation status is correlated with the cell latent so the
mutation-stratified analysis has something to chew on.

Usage:  python demo/make_demo_data.py [--force]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FEAT = PROC / "orgdrug"
EXT = ROOT / "data" / "external"

N_CELLS, N_DRUGS, N_GENES, SCF_DIM = 80, 30, 400, 512
N_PATH, N_MUT_FILL, N_TGT_POOL = 51, 96, 40

# 30 demonstration compounds: chemically valid, deliberately generic
# structures (the demo exercises the pipeline, not the chemistry).
# (name, SMILES, pathway class)
DRUGS = [
    ("DemoDrug01", "CC(=O)Oc1ccccc1C(=O)O", "pathway A"),
    ("DemoDrug02", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "pathway A"),
    ("DemoDrug03", "CC(=O)Nc1ccc(O)cc1", "pathway A"),
    ("DemoDrug04", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "pathway A"),
    ("DemoDrug05", "O=C(O)c1ccc(O)cc1", "pathway A"),
    ("DemoDrug06", "COC(=O)c1ccccc1", "pathway B"),
    ("DemoDrug07", "NC(=O)c1ccccc1", "pathway B"),
    ("DemoDrug08", "CNC(=O)c1ccccc1", "pathway B"),
    ("DemoDrug09", "O=C(Nc1ccccc1)c1ccccc1", "pathway B"),
    ("DemoDrug10", "COc1ccc(CCN)cc1", "pathway B"),
    ("DemoDrug11", "NS(=O)(=O)c1ccccc1", "pathway C"),
    ("DemoDrug12", "Cc1ccc(S(N)(=O)=O)cc1", "pathway C"),
    ("DemoDrug13", "O=S(=O)(N)c1ccc(Cl)cc1", "pathway C"),
    ("DemoDrug14", "COc1ccc(N)cc1", "pathway C"),
    ("DemoDrug15", "Clc1ccc(Cl)cc1", "pathway C"),
    ("DemoDrug16", "Fc1ccc(F)cc1", "pathway D"),
    ("DemoDrug17", "c1ccc(-c2ccccc2)cc1", "pathway D"),
    ("DemoDrug18", "c1ccncc1", "pathway D"),
    ("DemoDrug19", "c1ccc2ncccc2c1", "pathway D"),
    ("DemoDrug20", "O=C1CCCCC1", "pathway D"),
    ("DemoDrug21", "C1CCCCC1", "pathway E"),
    ("DemoDrug22", "Oc1ccccc1", "pathway E"),
    ("DemoDrug23", "Cc1ccccc1", "pathway E"),
    ("DemoDrug24", "Clc1ccccc1", "pathway E"),
    ("DemoDrug25", "Nc1ncnc2[nH]cnc12", "pathway E"),
    ("DemoDrug26", "NCC(=O)O", "pathway F"),
    ("DemoDrug27", "CCO", "pathway F"),
    ("DemoDrug28", "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O", "pathway F"),
    ("DemoDrug29", "COc1ccc(NC(=O)C)cc1", "pathway F"),
    ("DemoDrug30", "O=C(N)c1ccc(F)cc1", "pathway F"),
]

DRIVER_POOL = [
    "KRAS", "TP53", "BRAF", "EGFR", "ERBB2", "PIK3CA", "PTEN", "APC", "CTNNB1",
    "STK11", "KEAP1", "BAP1", "ARID1A", "SMARCA4", "CDKN2A", "RB1", "MYC",
    "NRAS", "HRAS", "MET", "ALK", "RET", "ROS1", "FGFR1", "FGFR2", "FGFR3",
    "JAK1", "JAK2", "STAT3", "IDH1", "IDH2", "TET2", "DNMT3A", "EZH2", "KMT2D",
    "NOTCH1", "SMAD4", "TGFBR2", "VHL", "SDHB", "BRCA1", "BRCA2", "ATM", "CHEK2",
    "PALB2", "CDK4", "CCND1", "CCNE1", "MDM2", "AKT1", "MTOR", "TSC1", "TSC2",
]

TISSUES = ["lung", "colon", "pancreas", "breast", "ovary", "gastric", "liver",
           "melanoma", "bladder", "prostate", "leukemia", "lymphoma"]


def build(seed=42):
    rng = np.random.default_rng(seed)
    PROC.mkdir(parents=True, exist_ok=True)
    FEAT.mkdir(parents=True, exist_ok=True)
    EXT.mkdir(parents=True, exist_ok=True)

    # ── entities ────────────────────────────────────────────────────────────
    cells = [f"SIDM{i:04d}" for i in range(N_CELLS)]
    drugs = [d[0] for d in DRUGS]

    # cell latent (4-d) drives gene / scf / pathway / KRAS views
    u = rng.normal(size=(N_CELLS, 4)).astype(np.float32)
    B_gene = rng.normal(size=(4, N_GENES)).astype(np.float32) * 0.5
    gene = u @ B_gene + rng.normal(size=(N_CELLS, N_GENES)).astype(np.float32) * 0.4
    gene = ((gene - gene.mean(0)) / gene.std(0)).astype(np.float32)   # z per gene
    gene_names = [f"LM{i:04d}" for i in range(N_GENES)]

    B2 = rng.normal(size=(4, SCF_DIM)).astype(np.float32) * 0.2
    scf = (rng.normal(size=(N_CELLS, SCF_DIM)).astype(np.float32) + u @ B2)

    # ── drug chemistry (Morgan at train time; MACCS precomputed here) ──────
    smiles_df = pd.DataFrame({"drug_name": drugs,
                              "smiles": [d[1] for d in DRUGS]})
    smiles_df.to_csv(EXT / "gdsc_drug_smiles.csv", index=False)

    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    maccs_rows = []
    for _, r in smiles_df.iterrows():
        mol = Chem.MolFromSmiles(r["smiles"])
        assert mol is not None, f"bad SMILES for {r.drug_name}"
        maccs_rows.append(np.array(MACCSkeys.GenMACCSKeys(mol),
                                   dtype=np.float32)[:167])
    maccs = np.stack(maccs_rows)
    np.save(FEAT / "maccs_matrix.npy", maccs)
    pd.DataFrame({"drug_name": drugs}).to_csv(FEAT / "maccs_drugs.csv", index=False)

    # drug latent lives in the fingerprint views (MACCS projection)
    W_v = rng.normal(size=(167, 4)).astype(np.float32)
    vd = np.tanh(maccs @ W_v)                                        # (30, 4)
    b_d = rng.normal(size=N_DRUGS).astype(np.float32) * 0.8          # drug bias

    # ── pair table (full grid) ─────────────────────────────────────────────
    rows = []
    for ci, c in enumerate(cells):
        for di, d in enumerate(drugs):
            rows.append((c, d, ci, di))
    meta = pd.DataFrame(rows, columns=["ModelID", "drug_name", "ci", "di"])
    meta["COSMIC_ID"] = 900000 + meta["ci"] * 31 + meta["di"]
    meta["tissue"] = [TISSUES[i % len(TISSUES)] for i in meta["ci"]]
    cos = meta.apply(
        lambda r: u[r.ci] @ vd[r.di], axis=1).values.astype(np.float32)
    kras = np.zeros(N_CELLS, dtype=bool)
    kras[u[:, 0] > 0.25] = True                       # KRAS ~ cell latent u0
    kras_eff = kras[meta["ci"].values] * vd[meta["di"].values, 0]
    y = (b_d[meta["di"].values] + 0.55 * cos / 2.0 + 0.35 * kras_eff
         + rng.normal(size=len(meta)).astype(np.float32) * 0.30)

    feats = gene[meta["ci"].values]                                   # pair rows
    pd.DataFrame(feats, columns=gene_names).to_parquet(
        PROC / "gdsc_cell_features.parquet", index=True)
    y_df = pd.DataFrame({"ln_IC50": y})
    y_df.to_parquet(PROC / "gdsc_response_lnic50.parquet", index=True)
    meta_out = meta.drop(columns=["ci", "di"])
    meta_out.to_parquet(PROC / "gdsc_metadata.parquet", index=True)

    np.save(PROC / "scfoundation_cell_emb.npy", scf)
    np.save(PROC / "scfoundation_cell_ids.npy", np.array(cells, dtype=object))

    # ── orgdrug/ feature tables ────────────────────────────────────────────
    # pathway scores: signal in a few terms, z-scored
    W_p = rng.normal(size=(4, N_PATH)).astype(np.float32) * 0.6
    W_p[:, 8:] *= 0.15                                                # most terms weak
    path = u @ W_p + rng.normal(size=(N_CELLS, N_PATH)).astype(np.float32) * 0.3
    path = ((path - path.mean(0)) / path.std(0)).astype(np.float32)
    np.save(FEAT / "pathway_scores.npy", path)
    np.save(FEAT / "pathway_cell_ids.npy", np.array(cells, dtype=object))
    pnames = [f"HALLMARK_TERM_{i:02d}" for i in range(N_PATH - 1)] + ["MICRO_Stemness"]
    pd.DataFrame({"pathway": pnames, "n_genes": 40}).to_csv(
        FEAT / "pathway_names.csv", index=False)

    # mutation matrix: real drivers (curated) + filler genes
    mut_genes = DRIVER_POOL + [f"FILLER{i:03d}" for i in range(N_MUT_FILL)]
    curated = [g in DRIVER_POOL for g in mut_genes]
    mut = np.zeros((N_CELLS, len(mut_genes)), dtype=np.float32)
    kr_i, tp_i, br_i = mut_genes.index("KRAS"), mut_genes.index("TP53"), mut_genes.index("BRAF")
    mut[kras, kr_i] = 1.0
    mut[rng.random(N_CELLS) < 0.45, tp_i] = 1.0
    mut[rng.random(N_CELLS) < 0.08, br_i] = 1.0
    filler = rng.random((N_CELLS, len(mut_genes))) < 0.05
    mut[filler & (mut == 0)] = 1.0
    np.save(FEAT / "mut_matrix.npy", mut)
    pd.DataFrame({"gene": mut_genes, "curated": curated}).to_csv(
        FEAT / "mut_genes.csv", index=False)
    np.save(FEAT / "mut_cell_ids.npy", np.array(cells, dtype=object))

    # drug targets + pathway one-hot
    tgt_pool = rng.choice(DRIVER_POOL, size=N_TGT_POOL, replace=False).tolist()
    tgt = np.zeros((N_DRUGS, len(tgt_pool)), dtype=np.float32)
    for d in range(N_DRUGS):
        for g in rng.choice(len(tgt_pool), size=rng.integers(2, 5), replace=False):
            tgt[d, g] = 1.0
    np.save(FEAT / "drug_target_matrix.npy", tgt)
    pd.DataFrame({"gene": tgt_pool}).to_csv(FEAT / "target_genes.csv", index=False)

    cats = sorted({d[2] for d in DRUGS})
    cat_i = {c: i for i, c in enumerate(cats)}
    onehot = np.zeros((N_DRUGS, len(cats)), dtype=np.float32)
    for di, d in enumerate(DRUGS):
        onehot[di, cat_i[d[2]]] = 1.0
    np.save(FEAT / "drug_pathway_onehot.npy", onehot)
    pd.DataFrame({"drug_name": drugs}).to_csv(FEAT / "drug_target_drugs.csv", index=False)
    pd.DataFrame({"pathway": cats}).to_csv(FEAT / "drug_pathway_names.csv", index=False)

    vocab = sorted(set(mut_genes) | set(tgt_pool))
    pd.DataFrame({"gene": vocab,
                  "vocab_idx": np.arange(1, len(vocab) + 1)}).to_csv(
        FEAT / "shared_gene_vocab.csv", index=False)

    print(f"demo dataset ready: {len(meta)} pairs, {N_CELLS} cells, "
          f"{N_DRUGS} drugs  (KRAS {kras.mean()*100:.0f}%)")
    print(f"  -> {PROC}")
    print(f"  -> {FEAT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.force or not (PROC / "gdsc_metadata.parquet").exists():
        build()
    else:
        print("demo data already present (use --force to rebuild)")
