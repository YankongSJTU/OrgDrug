# -*- coding: utf-8 -*-
"""
E5 — PRISM Data Preparation (v2, 2026-11-05)
=============================================
Builds the union pretraining dataset  GDSC-expanded + PRISM-19Q4-secondary
in the exact format of data/processed/expanded/ so that (the patched)
pretrain_orgdrug_expanded.py can consume it via --data-dir.

Sources (all on server, SHA256-verified mirror of PRISM Repurposing 19Q4 +
DepMap 24Q2, in data/raw/ccle/hf/):
  secondary_screen_params.csv   long table: broad_id, depmap_id, auc/ic50,
                                name, moa, target, smiles, phase
  OmicsCNGene.csv               (saved for the later CNA view; not used here)

Union dataset -> data/processed/expanded_prism/:
  X_gene / X_scf / X_fp / cell_idx / drug_idx / y / tr_mask / val_mask
  expanded_metadata.csv (+source column), cell_id_map.csv (**keyed by ModelID**),
  drug_id_map.csv, scf_avail.npy (0/1 per pair)

Also appends PRISM drugs (namespaced "PR::<broad_id>") to the shared feature
tables maccs/onehot/tgt (in-place, idempotent) so drug_set_rows() finds them.
Also writes expanded/scf_avail.npy for the bug-fix-only rerun.

Notes
  - y = ln(IC50 µM) for both sources (identical semantics to LN_IC50).
  - GDSC pairs reuse the original expanded/ X rows byte-for-byte; PRISM cell
    landmark z-scores are computed from the full-DepMap reference (BN absorbs
    the small scale difference).
  - FIXED cell map is keyed by ModelID: the original expanded/cell_id_map.csv
    is keyed by COSMIC_ID, which silently broke per-cell feature assembly in
    pretrain_orgdrug_expanded.py (all cells -> pair row 0; mut/path = 0).
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
RAW = ROOT / "data/raw/ccle"
HF = RAW / "hf"
EXP = ROOT / "data/processed/expanded"
OUT = ROOT / "data/processed/expanded_prism"
FEAT = ROOT / "data/processed/orgdrug"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass


def log(m):
    print(f"[prism-prep {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ─── 1. PRISM pairs ─────────────────────────────────────────────────────────

def build_prism_pairs():
    ss = pd.read_csv(HF / "secondary_screen_params.csv", low_memory=False)
    ic = pd.to_numeric(ss["ic50"], errors="coerce")
    keep = (ss["passed_str_profiling"] == True) & ic.notna() & (ic > 1e-3) & (ic < 1e4)
    ss = ss[keep].copy()
    ss["y"] = np.log(ic[keep])
    # replicate screens (screen_id MTS/HTS plates) -> median per (drug, cell)
    ss = ss.groupby(["broad_id", "depmap_id"]).agg(
        y=("y", "median"), smiles=("smiles", "first"), name=("name", "first"),
        target=("target", "first"), moa=("moa", "first")).reset_index()
    log(f"PRISM filtered+deduped: {len(ss):,} pairs  "
        f"cells={ss['depmap_id'].nunique()}  drugs={ss['broad_id'].nunique()}")
    return ss


# ─── 2. Cell features for PRISM cells ──────────────────────────────────────

def build_prism_cell_features(prism_cells):
    """landmark z (2000) + scf (3072) rows for PRISM cells, plus avail masks."""
    lm_genes = list(pd.read_parquet(ROOT / "data/processed/gdsc_cell_features.parquet").columns)
    log(f"landmark genes: {len(lm_genes)}")

    t0 = time.time()
    expr = pd.read_csv(RAW / "OmicsExpression_TPM.csv", index_col=0)
    if "IsDefaultEntryForModel" in expr:
        expr = expr[expr["IsDefaultEntryForModel"] == "Yes"]
    genes = {c: c.split(" (")[0].strip() for c in expr.columns
             if re.match(r"^[A-Za-z0-9._-]+ \(\d+\)$", c)}
    e = expr[list(genes)].rename(columns=genes)
    e.index = expr["ModelID"].values if "ModelID" in expr else e.index
    e = e[~e.index.duplicated(keep="first")]
    e = np.log2(e.astype(np.float32) + 1.0)
    mu, sd = e.mean(axis=0), e.std(axis=0).replace(0, 1.0)
    e = ((e - mu) / sd)
    log(f"expression z-matrix {e.shape} ({time.time()-t0:.0f}s); "
        f"PRISM cells present: {len(set(e.index) & set(prism_cells))}/{len(prism_cells)}")

    have = [c for c in prism_cells if c in e.index]
    Xg = np.zeros((len(prism_cells), 2000), dtype=np.float32)
    pos = {c: i for i, c in enumerate(prism_cells)}
    lm_common = [g for g in lm_genes if g in e.columns]
    for c in have:
        row = e.loc[c, lm_common]
        Xg[pos[c], :len(lm_common)] = row.values.astype(np.float32)
    gene_avail = np.zeros(len(prism_cells), dtype=np.float32)
    for c in have:
        gene_avail[pos[c]] = 1.0

    scf = np.load(ROOT / "data/processed/scfoundation_cell_emb.npy")
    scf_ids = np.load(ROOT / "data/processed/scfoundation_cell_ids.npy", allow_pickle=True)
    smap = {c: scf[i] for i, c in enumerate(scf_ids.tolist())}
    Xs = np.zeros((len(prism_cells), scf.shape[1]), dtype=np.float32)
    scf_avail = np.zeros(len(prism_cells), dtype=np.float32)
    for c in prism_cells:
        if c in smap:
            Xs[pos[c]] = smap[c].astype(np.float32)
            scf_avail[pos[c]] = 1.0
    log(f"cell feats: gene_avail={gene_avail.sum():.0f}/{len(prism_cells)}  "
        f"scf_avail={scf_avail.sum():.0f}/{len(prism_cells)}")
    return Xg, Xs, gene_avail, scf_avail


# ─── 3. PRISM drug rows in the shared tables ────────────────────────────────

def _clean_smiles(s):
    """PRISM smiles fields are comma-separated lists (salt/stereoisomer forms)
    -> keep the longest fragment (the active compound)."""
    if not isinstance(s, str) or s.strip() in ("", ","):
        return None
    frags = [f.strip() for f in s.split(",") if len(f.strip()) >= 10]
    if frags:
        return max(frags, key=len)
    t = s.strip().rstrip(",").strip()
    return t or None


def append_drug_tables(prism):
    """Append PR::<broad_id> rows to maccs / onehot(Unclassified) / tgt tables."""
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, rdMolDescriptors

    drugs = prism.drop_duplicates("broad_id")[["broad_id", "smiles", "target"]].copy()
    drugs["key"] = "PR::" + drugs["broad_id"]

    # --- maccs (skip if already appended) ---
    mac_names = pd.read_csv(FEAT / "maccs_drugs.csv")["drug_name"].tolist()
    todo = drugs[~drugs["key"].isin(mac_names)]
    if len(todo):
        rows, names = [], []
        for _, r in todo.iterrows():
            smi = _clean_smiles(r["smiles"])
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                continue
            fp = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)[:167]
            rows.append(fp)
            names.append(r["key"])
        if rows:
            old = np.load(FEAT / "maccs_matrix.npy")
            np.save(FEAT / "maccs_matrix.npy",
                    np.concatenate([old, np.stack(rows)]).astype(np.float32))
            pd.DataFrame({"drug_name": mac_names + names}).to_csv(
                FEAT / "maccs_drugs.csv", index=False)
        log(f"maccs appended {len(names)}/{len(todo)}")

    # --- onehot -> Unclassified ---
    onehot = np.load(FEAT / "drug_pathway_onehot.npy")
    oh_names = pd.read_csv(FEAT / "drug_target_drugs.csv")["drug_name"].tolist()
    pw = pd.read_csv(FEAT / "drug_pathway_names.csv")
    ucol = int(pw.index[pw["pathway"] == "Unclassified"][0])
    todo = drugs[~drugs["key"].isin(oh_names)]
    if len(todo):
        add = np.zeros((len(todo), onehot.shape[1]), dtype=np.float32)
        add[:, ucol] = 1.0
        np.save(FEAT / "drug_pathway_onehot.npy", np.concatenate([onehot, add]))
        pd.DataFrame({"drug_name": oh_names + todo["key"].tolist()}).to_csv(
            FEAT / "drug_target_drugs.csv", index=False)
        log(f"onehot appended {len(todo)} (Unclassified)")

    # --- tgt matrix: PRISM target text -> existing target genes only ---
    tgt = np.load(FEAT / "drug_target_matrix.npy")
    tgt_genes = pd.read_csv(FEAT / "target_genes.csv")["gene"].tolist()
    tg_names = oh_names  # same key file (drug_target_drugs.csv)
    tg_set = set(tgt_genes)
    todo = drugs[~drugs["key"].isin(tg_names)]
    if len(todo):
        add = np.zeros((len(todo), len(tgt_genes)), dtype=np.float32)
        gi = {g: i for i, g in enumerate(tgt_genes)}
        n_ann = 0
        for r_i, (_, r) in enumerate(todo.iterrows()):
            if not isinstance(r["target"], str) or r["target"] in ("NA", ""):
                continue
            syms = {re.sub(r"\s*\(.*\)$", "", t.strip().upper())
                    for t in re.split(r"[;,|]", r["target"])}
            hit = [g for g in syms if g in tg_set]
            for g in hit:
                add[r_i, gi[g]] = 1.0
            n_ann += bool(hit)
        np.save(FEAT / "drug_target_matrix.npy", np.concatenate([tgt, add]))
        pd.DataFrame({"drug_name": tg_names + todo["key"].tolist()}).to_csv(
            FEAT / "drug_target_drugs.csv", index=False)
        log(f"tgt appended {len(todo)} rows, {n_ann} with >=1 known target gene")
    return drugs


# ─── 4. PRISM fingerprints ──────────────────────────────────────────────────

def _morgan2048(mol):
    """Morgan radius-2 2048-bit fp, robust to rdkit API generation."""
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    try:
        from rdkit.Chem import rdMolDescriptors
        gen = rdMolDescriptors.GetMorganGenerator(radius=2, fpSize=2048)
        return gen.GetFingerprintAsNumPy(mol).astype(np.float32)
    except AttributeError:  # older rdkit
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr.astype(np.float32)


def build_prism_fp(prism):
    """Morgan 2048 for PRISM pairs (MACCS/onehot come from the shared tables)."""
    from rdkit import Chem
    drugs = prism.drop_duplicates("broad_id")[["broad_id", "smiles"]]
    mol_ok = {}
    for _, r in drugs.iterrows():
        smi = _clean_smiles(r["smiles"])
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        mol_ok[r["broad_id"]] = _morgan2048(mol)
    log(f"morgan ok for {len(mol_ok)}/{drugs['broad_id'].nunique()} PRISM drugs")
    return mol_ok


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    prism = build_prism_pairs()

    prism_cells = sorted(prism["depmap_id"].unique())
    Xg_p, Xs_p, gene_avail_c, scf_avail_c = build_prism_cell_features(prism_cells)
    drugs_tbl = append_drug_tables(prism)
    morgan = build_prism_fp(prism)

    # keep only pairs whose drug got a valid morgan fp
    prism = prism[prism["broad_id"].isin(morgan)]

    # ── base GDSC expanded ──
    log("loading GDSC expanded ...")
    Xg_g = np.load(EXP / "X_gene.npy").astype(np.float32)
    Xs_g = np.load(EXP / "X_scf.npy").astype(np.float32)
    Xf_g = np.load(EXP / "X_fp.npy").astype(np.float32)
    ci_g = np.load(EXP / "cell_idx.npy").astype(np.int64)
    di_g = np.load(EXP / "drug_idx.npy").astype(np.int64)
    y_g = np.load(EXP / "y.npy").astype(np.float32)
    tr_g = np.load(EXP / "tr_mask.npy").astype(bool)
    va_g = np.load(EXP / "val_mask.npy").astype(bool)
    meta_g = pd.read_csv(EXP / "expanded_metadata.csv")
    cm = pd.read_csv(EXP / "cell_id_map.csv")
    dm = pd.read_csv(EXP / "drug_id_map.csv")
    n_cell_g, n_drug_g = len(cm), len(dm)

    # expanded/scf_avail sidecar for the bug-fix-only rerun (from has_scf)
    c2i = {str(r[cm.columns[0]]): int(r[cm.columns[1]]) for _, r in cm.iterrows()}
    meta_cos = meta_g.drop_duplicates("COSMIC_ID").set_index("COSMIC_ID")["has_scf"]
    meta_cos.index = meta_cos.index.astype(str)  # COSMIC ids: str keys vs int64 index
    cell_has_scf = np.zeros(n_cell_g + 1, dtype=np.float32)
    for c, i in c2i.items():
        cell_has_scf[i] = float(meta_cos.get(c, False))
    np.save(EXP / "scf_avail.npy", cell_has_scf[ci_g])
    log("wrote expanded/scf_avail.npy "
        f"({(cell_has_scf[ci_g] > 0).mean()*100:.1f}% of pairs have scf)")

    # ── PRISM entity maps (continuing the numbering) ──
    cell_pos = {c: i for i, c in enumerate(prism_cells)}
    prism_keys = "PR::" + prism["broad_id"].values
    new_cells = sorted(set(prism_cells))
    cell2idx = {c: n_cell_g + 1 + i for i, c in enumerate(new_cells)}
    d_keys = sorted(set(prism_keys.tolist()))
    drug2idx = {d: n_drug_g + 1 + i for i, d in enumerate(d_keys)}
    ci_p = prism["depmap_id"].map(cell2idx).values.astype(np.int64)
    di_p = prism["broad_id"].map(lambda b: drug2idx["PR::" + b]).values.astype(np.int64)
    y_p = prism["y"].values.astype(np.float32)

    # ── PRISM X rows ──
    fp_dim = Xf_g.shape[1]
    Xg_rows = Xg_p[[cell_pos[c] for c in prism["depmap_id"]]]
    Xs_rows = Xs_p[[cell_pos[c] for c in prism["depmap_id"]]]
    Xf_rows = np.zeros((len(prism), fp_dim), dtype=np.float32)
    for i, b in enumerate(prism["broad_id"].values):
        Xf_rows[i, :2048] = morgan[b]

    # ── union ──
    log("concatenating union dataset ...")
    X_gene = np.concatenate([Xg_g, Xg_rows])
    X_scf = np.concatenate([Xs_g, Xs_rows])
    X_fp = np.concatenate([Xf_g, Xf_rows])
    cell_idx = np.concatenate([ci_g, ci_p])
    drug_idx = np.concatenate([di_g, di_p])
    y = np.concatenate([y_g, y_p])
    n_all = len(y)

    # PRISM val = 3% random pairs (seeded); GDSC masks unchanged
    rng = np.random.RandomState(42)
    va_p = rng.rand(len(y_p)) < 0.03
    val_mask = np.concatenate([va_g, va_p])
    tr_mask = np.concatenate([tr_g, ~va_p])
    log(f"union: {n_all:,} pairs  train={tr_mask.sum():,}  val={val_mask.sum():,}")

    # scf avail per pair
    scf_avail_p = scf_avail_c[[cell_pos[c] for c in prism["depmap_id"]]]
    scf_avail = np.concatenate([cell_has_scf[ci_g], scf_avail_p]).astype(np.float32)

    meta_p = pd.DataFrame({
        "COSMIC_ID": "PRISM", "CELL_LINE_NAME": prism["name"].values,
        "DRUG_NAME": prism_keys, "LN_IC50": y_p, "ModelID": prism["depmap_id"].values,
        "is_train": ~va_p, "has_scf": scf_avail_p > 0, "source": "prism"})
    meta_g2 = meta_g.rename(columns={"DRUG_NAME": "DRUG_NAME"})
    meta_g2["source"] = "gdsc"
    meta_u = pd.concat([meta_g2, meta_p], ignore_index=True)

    log("saving npys (this is the big IO step) ...")
    np.save(OUT / "X_gene.npy", X_gene)
    np.save(OUT / "X_scf.npy", X_scf)
    np.save(OUT / "X_fp.npy", X_fp)
    np.save(OUT / "cell_idx.npy", cell_idx)
    np.save(OUT / "drug_idx.npy", drug_idx)
    np.save(OUT / "y.npy", y)
    np.save(OUT / "tr_mask.npy", tr_mask)
    np.save(OUT / "val_mask.npy", val_mask)
    np.save(OUT / "scf_avail.npy", scf_avail)
    meta_u.to_csv(OUT / "expanded_metadata.csv", index=False)

    # **ModelID-keyed** cell map (fixes the COSMIC/ModelID assembly bug)
    cos2model = meta_g.drop_duplicates("COSMIC_ID").set_index("COSMIC_ID")["ModelID"]
    pd.DataFrame({"ModelID": [cos2model.get(str(c), c) for c in cm[cm.columns[0]]] + new_cells,
                  "cell_idx": list(cm[cm.columns[1]]) + list(range(n_cell_g + 1, n_cell_g + 1 + len(new_cells)))
                  }).to_csv(OUT / "cell_id_map.csv", index=False)
    pd.DataFrame({"DRUG_NAME": dm[dm.columns[0]].tolist() + d_keys,
                  "drug_idx": dm[dm.columns[1]].tolist() + list(range(n_drug_g + 1, n_drug_g + 1 + len(d_keys)))
                  }).to_csv(OUT / "drug_id_map.csv", index=False)

    log(f"PRISM pairs final: {len(y_p):,}  cells +{len(new_cells)}  drugs +{len(d_keys)}")
    log(f"y gdsc mean={y_g.mean():.2f}±{y_g.std():.2f} | prism mean={y_p.mean():.2f}±{y_p.std():.2f}")
    log(f"ALL DONE {(time.time()-t0)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
