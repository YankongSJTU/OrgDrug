# -*- coding: utf-8 -*-
"""
OrgDrug Feature Preparation
============================
Builds the new modality features on top of the existing GDRNet data:

  1. Mutation binary matrix  (driver-gene set, from DepMap damaging MAF matrix)
  2. Pathway activity scores (50 MSigDB Hallmark terms + stemness panel,
                              mean-z over member genes from full-transcriptome TPM)
  3. Drug target matrix      (GDSC PUTATIVE_TARGET -> gene-symbol binary vector)
                             + PATHWAY_NAME one-hot
  4. MACCS keys              (167-bit, RDKit)
  5. Shared gene vocabulary  (union of driver genes & target genes -> one
                              embedding table shared by mut & tgt views)

Outputs -> data/processed/orgdrug/
Alignment: features are stored keyed by ModelID / drug_name; the training
script aligns them to the LCLO sample table by name.
Idempotent: each block is skipped if its output already exists (use --force).
"""

import os
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ORGDRUG_ROOT",
                 Path(__file__).resolve().parents[1]))
RAW = ROOT / "data/raw"
PROC = ROOT / "data/processed"
OUT = PROC / "orgdrug"
OUT.mkdir(parents=True, exist_ok=True)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass

# ─── Curated driver gene panel (~140 core cancer drivers) ───────────────────
CURATED_DRIVERS = [
    # RTK / RAS / MAPK
    "EGFR", "ERBB2", "ERBB3", "ERBB4", "ALK", "ROS1", "RET", "MET", "NTRK1",
    "NTRK2", "NTRK3", "FGFR1", "FGFR2", "FGFR3", "FGFR4", "PDGFRA", "PDGFRB",
    "KIT", "FLT3", "FLT4", "KRAS", "NRAS", "HRAS", "BRAF", "ARAF", "RAF1",
    "MAP2K1", "MAP2K2", "MAPK1", "MAP3K1", "RAC1", "RIT1", "RHOA", "PTPN11",
    # PI3K / mTOR
    "PIK3CA", "PIK3CB", "PIK3R1", "PTEN", "AKT1", "AKT2", "AKT3", "MTOR",
    "TSC1", "TSC2", "STK11", "KEAP1", "NFE2L2", "RPTOR",
    # Tumor suppressors / genome stability
    "TP53", "RB1", "APC", "CTNNB1", "AXIN1", "AXIN2", "SMAD2", "SMAD3",
    "SMAD4", "TGFBR2", "CDC73", "MEN1", "VHL", "SDHB", "SDHD", "EPAS1",
    "HIF1A", "MUTYH", "CHEK2", "ATM", "BRCA1", "BRCA2", "PALB2", "RAD51D",
    # Epigenetic regulators
    "IDH1", "IDH2", "TET2", "DNMT3A", "DNMT3B", "ASXL1", "ASXL2", "EZH2",
    "KMT2D", "KMT2C", "SETD2", "KDM6A", "BAP1", "PBRM1", "ARID1A", "ARID1B",
    "ARID2", "SMARCA4", "SMARCB1", "CREBBP", "EP300", "NSD1", "NSD3",
    # Cell cycle
    "CDKN2A", "CDKN2B", "CDK4", "CCND1", "CCNE1", "CCND3", "CDK6", "MDM2",
    "MDM4", "CDK12",
    # Apoptosis / NF-kB / JAK-STAT
    "BCL2", "BCL2L1", "MCL1", "TNFAIP3", "TRAF3", "CYLD", "MYD88", "CD79A",
    "CD79B", "CARD11", "JAK1", "JAK2", "JAK3", "STAT3", "STAT5B", "CSF1R",
    "ABL1", "ABL2", "CSF3R", "GNAS", "MPL", "CALR", "CXCR4",
    # Splicing / hematopoietic
    "SF3B1", "SRSF2", "U2AF1", "ZRSR2", "RUNX1", "ETNK1", "GATA1", "GATA2",
    "CEBPA", "CBL", "SETBP1", "SRSF1",
    # NOTCH / Wnt / Hedgehog / other OG
    "NOTCH1", "NOTCH2", "NOTCH3", "MYC", "MYCN", "MYCL", "GNAQ", "GNA11",
    "PLCB4", "CYSLTR2", "TERT", "TERC", "MED12", "GRM3", "H3F3A", "PPM1D",
]

# Stemness marker panel (curated heuristic: embryonic + CSC + Wnt-stem markers)
STEMNESS_PANEL = [
    "SOX2", "POU5F1", "NANOG", "KLF4", "MYC", "LIN28A", "LIN28B",
    "LGR5", "ASCL2", "PROM1", "ALDH1A1", "BMI1", "MSI1", "NES",
    "CD44", "ITGA6", "EPCAM",
]

HALLMARK_URL = ("https://maayanlab.cloud/Enrichr/geneSetLibrary"
                "?mode=json&libraryName=MSigDB_Hallmark_2020")
N_DRIVER_MAX = 448


def _sym(col: str) -> str:
    """'TSPAN6 (7105)' -> 'TSPAN6'"""
    return col.split(" (")[0].strip()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── 1. Hallmark gene sets ──────────────────────────────────────────────────

def fetch_hallmark():
    gs_path = OUT / "hallmark_genesets.json"
    if gs_path.exists():
        return json.loads(gs_path.read_text())
    import urllib.request
    try:
        log("downloading Hallmark gene sets from Enrichr ...")
        urllib.request.urlretrieve(HALLMARK_URL, gs_path)
    except Exception as e:  # offline fallback: earlier manual fetch
        log(f"download failed ({e}); trying /tmp/hallmark.json")
        import shutil
        shutil.copy("/tmp/hallmark.json", gs_path)
    data = json.loads(gs_path.read_text())
    terms = data["MSigDB_Hallmark_2020"]["terms"]
    # keep only gene list, drop the 1.0 weights
    return {k: sorted(v.keys()) for k, v in terms.items()}


# ─── 2. Expression matrix (cells x genes, log2TPM+1 z-scored) ───────────────

_expr_cache = None

def load_expression():
    global _expr_cache
    if _expr_cache is not None:
        return _expr_cache
    log("loading OmicsExpression_TPM.csv (this takes a few minutes) ...")
    t0 = time.time()
    df = pd.read_csv(RAW / "ccle/OmicsExpression_TPM.csv", index_col=0)
    df = df[df["IsDefaultEntryForModel"] == "Yes"] if "IsDefaultEntryForModel" in df else df
    genes = {c: _sym(c) for c in df.columns if re.match(r"^[A-Za-z0-9._-]+ \(\d+\)$", c)}
    expr = df[list(genes)].rename(columns=genes)
    expr.index = df["ModelID"] if "ModelID" in df else df.index
    expr = np.log2(expr.astype(np.float32) + 1.0)
    expr = expr.T  # genes x cells
    # z-score per gene across cells
    mu = expr.mean(axis=1)
    sd = expr.std(axis=1).replace(0, 1.0)
    expr = ((expr.sub(mu, axis=0)).div(sd, axis=0)).T  # cells x genes
    log(f"expression z-matrix: {expr.shape}  ({time.time()-t0:.0f}s)")
    _expr_cache = expr
    return expr


# ─── 3. Pathway / microenvironment scores ───────────────────────────────────

def build_pathway(genesets, stem_panel):
    out_scores = OUT / "pathway_scores.npy"
    out_names = OUT / "pathway_names.csv"
    out_cells = OUT / "pathway_cell_ids.npy"
    if out_scores.exists():
        log("pathway scores exist, skipping")
        return
    expr = load_expression()
    terms = {}
    for name, genes in genesets.items():
        present = [g for g in genes if g in expr.columns]
        if len(present) >= 5:
            terms["HALLMARK_" + name] = present
        else:
            log(f"  skip {name}: only {len(present)} genes present")
    present_stem = [g for g in stem_panel if g in expr.columns]
    terms["MICRO_Stemness"] = present_stem
    log(f"computing {len(terms)} pathway mean-z scores ...")
    mat = np.stack([expr[genes].mean(axis=1).values for genes in terms.values()], axis=1)
    np.save(out_scores, mat.astype(np.float32))
    pd.DataFrame({"pathway": list(terms.keys()),
                  "n_genes": [len(g) for g in terms.values()]}).to_csv(out_names, index=False)
    np.save(out_cells, expr.index.values.astype(str))
    log(f"pathway scores: {mat.shape} -> {out_scores.name}")


# ─── 4. Mutation binary matrix over driver genes ────────────────────────────

def build_mutation():
    out_mat = OUT / "mut_matrix.npy"
    if out_mat.exists():
        log("mutation matrix exists, skipping")
        return
    log("loading damaging mutation matrix ...")
    t0 = time.time()
    df = pd.read_csv(RAW / "ccle/OmicsSomaticMutations_damaging.csv", index_col=0)
    df.index = df["ModelID"].values
    gene_cols = {c: _sym(c) for c in df.columns if re.match(r"^[A-Za-z0-9._-]+ \(\d+\)$", c)}
    mat_all = df[list(gene_cols)].rename(columns=gene_cols).astype(np.float32)
    log(f"damaging matrix loaded: {mat_all.shape} ({time.time()-t0:.0f}s)")

    freq = mat_all.sum(axis=0).sort_values(ascending=False)
    drivers = [g for g in CURATED_DRIVERS if g in mat_all.columns]
    fill = [g for g in freq.index if g not in set(drivers)]
    n_fill = max(0, N_DRIVER_MAX - len(drivers))
    drivers = drivers + fill[:n_fill]
    log(f"driver set: {len(drivers)} genes ({len(CURATED_DRIVERS)} curated matched "
        f"+ {len(drivers)-len([g for g in CURATED_DRIVERS if g in mat_all.columns])} freq-filled)")

    sub = mat_all[drivers]
    np.save(out_mat, sub.values.astype(np.float32))
    pd.DataFrame({"gene": drivers,
                  "curated": [g in CURATED_DRIVERS for g in drivers]}).to_csv(
        OUT / "mut_genes.csv", index=False)
    np.save(OUT / "mut_cell_ids.npy", sub.index.values.astype(str))
    log(f"mutation matrix: {sub.shape}, mean mut/cell={sub.values.sum(1).mean():.1f}")


# ─── 5. Drug target matrix + pathway one-hot ────────────────────────────────

def build_drug_features(smiles_df):
    out_tgt = OUT / "drug_target_matrix.npy"
    if out_tgt.exists():
        log("drug features exist, skipping")
        return
    # --- collect GDSC drug annotations (dedupe by DRUG_NAME) ---
    ann = pd.concat([
        pd.read_csv(RAW / "gdsc/GDSC2_fitted_dose_response.csv",
                    usecols=["DRUG_NAME", "PUTATIVE_TARGET", "PATHWAY_NAME"], low_memory=False),
        pd.read_csv(RAW / "gdsc/GDSC1_fitted_dose_response.csv",
                    usecols=["DRUG_NAME", "PUTATIVE_TARGET", "PATHWAY_NAME"], low_memory=False),
    ]).drop_duplicates("DRUG_NAME")
    ann = ann.set_index("DRUG_NAME")

    expr = load_expression()
    valid_syms = set(expr.columns)

    # --- target text -> symbol list ---
    def parse_targets(txt):
        if not isinstance(txt, str) or not txt.strip():
            return []
        toks = re.split(r"[;,|]", txt)
        out = []
        for t in toks:
            t = t.strip().upper()
            t = re.sub(r"\s*\(.*\)$", "", t)
            if t in valid_syms:
                out.append(t)
        return sorted(set(out))

    drugs = sorted(smiles_df["drug_name"].unique())
    target_sets, pathway_cats = {}, {}
    for d in drugs:
        if d not in ann.index:
            target_sets[d], pathway_cats[d] = [], "Unknown"
            continue
        target_sets[d] = parse_targets(ann.loc[d, "PUTATIVE_TARGET"])
        pw = ann.loc[d, "PATHWAY_NAME"]
        pathway_cats[d] = pw if isinstance(pw, str) else "Unknown"

    target_genes = sorted({g for v in target_sets.values() for g in v})
    tg_index = {g: i for i, g in enumerate(target_genes)}
    pw_names = sorted(set(pathway_cats.values()))
    pw_index = {p: i for i, p in enumerate(pw_names)}

    T = np.zeros((len(drugs), len(target_genes)), dtype=np.float32)
    P = np.zeros((len(drugs), len(pw_names)), dtype=np.float32)
    for r, d in enumerate(drugs):
        for g in target_sets[d]:
            T[r, tg_index[g]] = 1.0
        P[r, pw_index[pathway_cats[d]]] = 1.0

    np.save(out_tgt, T)
    pd.DataFrame({"gene": target_genes}).to_csv(OUT / "target_genes.csv", index=False)
    pd.DataFrame({"drug_name": drugs}).to_csv(OUT / "drug_target_drugs.csv", index=False)
    np.save(OUT / "drug_pathway_onehot.npy", P)
    pd.DataFrame({"pathway": pw_names}).to_csv(OUT / "drug_pathway_names.csv", index=False)
    n_annot = sum(1 for d in drugs if target_sets[d])
    log(f"drug targets: {len(drugs)} drugs, {n_annot} annotated, "
        f"{len(target_genes)} target genes, {len(pw_names)} pathway cats")


# ─── 6. MACCS keys ──────────────────────────────────────────────────────────

def build_maccs(smiles_df):
    out = OUT / "maccs_matrix.npy"
    if out.exists():
        log("MACCS exists, skipping")
        return
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    rows, names = [], []
    for _, r in smiles_df.iterrows():
        mol = Chem.MolFromSmiles(str(r["smiles"]))
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        rows.append(np.array(fp, dtype=np.float32)[:167])
        names.append(r["drug_name"])
    np.save(out, np.stack(rows))
    pd.DataFrame({"drug_name": names}).to_csv(OUT / "maccs_drugs.csv", index=False)
    log(f"MACCS: {len(names)} drugs")


# ─── 7. Shared gene vocabulary (mut ∪ target) ──────────────────────────────

def build_shared_vocab():
    out = OUT / "shared_gene_vocab.csv"
    if out.exists():
        log("shared vocab exists, skipping")
        return
    mut_genes = pd.read_csv(OUT / "mut_genes.csv")["gene"].tolist()
    tgt_genes = pd.read_csv(OUT / "target_genes.csv")["gene"].tolist()
    shared = sorted(set(mut_genes) | set(tgt_genes))
    pd.DataFrame({"gene": shared, "vocab_idx": np.arange(1, len(shared) + 1)}).to_csv(out, index=False)
    log(f"shared gene vocab: {len(shared)} (mut {len(mut_genes)} ∪ tgt {len(tgt_genes)})")


def main():
    log("── OrgDrug feature preparation ──")
    genesets = fetch_hallmark()
    log(f"hallmark terms: {len(genesets)}")
    (OUT / "stemness_panel.json").write_text(json.dumps(STEMNESS_PANEL))

    build_pathway(genesets, STEMNESS_PANEL)
    build_mutation()

    smiles_df = pd.read_csv(ROOT / "data/external/gdsc_drug_smiles.csv")
    build_drug_features(smiles_df)
    build_maccs(smiles_df)
    build_shared_vocab()

    log("ALL DONE. outputs:")
    for f in sorted(OUT.glob("*")):
        if f.suffix in (".npy", ".csv", ".json"):
            log(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
