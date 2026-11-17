# -*- coding: utf-8 -*-
"""
LCLO Extra Baselines — MOLI + DeepTTA
=========================================
Runs MOLI and DeepTTA under the identical five-fold LCLO protocol
(700 cell lines / 229 drugs / seed 42) used for all baselines in
table2_gdsc_lclo.csv, then emits the same metric format so rows can be
merged into Table 1 of the OrgDrug manuscript.

Usage (from GDRNet/src, via run_py.sh):
  /export/home/kongyan/run_py.sh rerun_lclo_extra.py --gpu 0
"""

import os
import argparse
import io
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("ORGDRUG_ROOT",
                 Path(__file__).resolve().parents[1]))
FEAT = ROOT / "data/processed/orgdrug"
EXT = ROOT / "data/external"
OUT_DIR = ROOT / "results/lclo_extra"
TABLES = ROOT / "results/tables"
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--patience", type=int, default=40)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--models", type=str, default="MOLI,DeepTTA")
args = parser.parse_args()

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
DEVICE = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"


def load_omics_views(cell_ids):
    """Join per-cell mutation (448) and CNA matrices by cell id. Zero-fill misses."""
    out = {}
    for name, mat_file, id_file in [
        ("mut", "mut_matrix.npy", "mut_cell_ids.npy"),
        ("cna", "cna_matrix.npy", "cna_cell_ids.npy"),
    ]:
        try:
            mat = np.load(FEAT / mat_file)
            ids = np.load(FEAT / id_file, allow_pickle=True)
            row = {str(c): i for i, c in enumerate(ids)}
            X = np.zeros((len(cell_ids), mat.shape[1]), dtype=np.float32)
            hit = 0
            for j, c in enumerate(map(str, cell_ids)):
                if c in row:
                    X[j] = mat[row[c]]
                    hit += 1
            print(f"  {name} view: {mat.shape[1]} dims, matched {hit}/{len(cell_ids)} cells",
                  flush=True)
            out[name] = X
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] {name} view unavailable ({e}); zero-filled", flush=True)
            out[name] = np.zeros((len(cell_ids), 1), dtype=np.float32)
    return out


def main():
    from data_loader import load_gdsc_lclo
    from eval_utils import compute_full_eval, aggregate_fold_results, format_results_table
    from models.moli import MOLI, MOLIDataset, train_moli
    from models.deeptta import DeepTTA, DeepTTADataset, train_deeptta, build_vocab, encode_smiles

    t0 = time.time()
    data = load_gdsc_lclo(n_splits=5, seed=42)

    # ── per-pair omics views (mutation + CNA) for MOLI ──
    print("Loading mutation/CNA views for MOLI ...", flush=True)
    omics = load_omics_views(data.cell_ids)
    n_mut = omics["mut"].shape[1]
    n_cna = omics["cna"].shape[1]

    # ── SMILES table for DeepTTA ──
    smiles_df = pd.read_csv(EXT / "gdsc_drug_smiles.csv")
    smi_by_drug = dict(zip(smiles_df["drug_name"].astype(str),
                           smiles_df["smiles"].astype(str)))
    all_smiles = [s for s in smi_by_drug.values() if s and s.lower() != "nan"]
    vocab = build_vocab(all_smiles)
    print(f"DeepTTA SMILES vocab: {len(vocab)} tokens over {len(all_smiles)} drugs",
          flush=True)
    smi_ids_by_drug = {d: encode_smiles(s, vocab) for d, s in smi_by_drug.items()
                       if s and s.lower() != "nan"}

    # per-pair SMILES ids (fallback pad for missing)
    pad_ids = np.zeros((1, 120), dtype=np.int64)
    X_smi = np.stack([smi_ids_by_drug.get(str(d), pad_ids[0]) for d in data.drug_names])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    models_to_run = [m.strip() for m in args.models.split(",")]
    all_results, all_preds = {}, {}

    for model_name in models_to_run:
        fold_evals, fold_pred_rows = [], []
        for fold_idx, (tr_mask, te_mask) in enumerate(data.folds):
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            tf = time.time()
            y_te = data.y[te_mask]
            drug_te, cell_te = data.drug_names[te_mask], data.cell_ids[te_mask]

            if model_name == "MOLI":
                model = MOLI(n_genes=2000, n_mut=n_mut, n_cna=n_cna, fp_bits=2048)
                tr_ds = MOLIDataset(data.X_gene[tr_mask], omics["mut"][tr_mask],
                                    omics["cna"][tr_mask], data.X_fp[tr_mask],
                                    data.y[tr_mask])
                te_ds = MOLIDataset(data.X_gene[te_mask], omics["mut"][te_mask],
                                    omics["cna"][te_mask], data.X_fp[te_mask], y_te)
                _, preds = train_moli(model, tr_ds, te_ds, n_epochs=args.epochs,
                                      batch_size=args.batch, lr=args.lr,
                                      patience=args.patience, device=DEVICE)
            elif model_name == "DeepTTA":
                model = DeepTTA(n_genes=2000, vocab_size=len(vocab))
                tr_ds = DeepTTADataset(data.X_gene[tr_mask], X_smi[tr_mask], data.y[tr_mask])
                te_ds = DeepTTADataset(data.X_gene[te_mask], X_smi[te_mask], y_te)
                _, preds = train_deeptta(model, tr_ds, te_ds, n_epochs=args.epochs,
                                         batch_size=args.batch, lr=args.lr,
                                         patience=args.patience, device=DEVICE)
            else:
                raise ValueError(model_name)

            ev = compute_full_eval(y_te, preds, drug_te, cell_te)
            fold_evals.append(ev)
            fold_pred_rows.append(pd.DataFrame({
                "drug_name": drug_te, "cell_id": cell_te,
                "y_true": y_te, "y_pred": preds, "fold": fold_idx}))
            print(f"[{model_name}] fold {fold_idx + 1}/5  "
                  f"Pooled_r={ev['pooled_pearson']:.4f}  "
                  f"PerDrug={ev['per_drug']['mean']:.4f}  "
                  f"AUROC30={ev['auroc_30']:.4f}  ({time.time() - tf:.0f}s)", flush=True)

        agg = aggregate_fold_results(fold_evals)
        all_results[model_name] = agg
        pd.concat(fold_pred_rows).to_csv(OUT_DIR / f"lclo_preds_{model_name}.csv",
                                         index=False)

    # ── Table-2-format output ──
    table = format_results_table(all_results)
    table.to_csv(TABLES / "lclo_extra_moli_deeptta.csv", index=False)
    print("\nSaved:", TABLES / "lclo_extra_moli_deeptta.csv", flush=True)
    cols = ["Pooled_Pearson", "PerDrug_Mean_r", "AUROC_30"]
    print(table[[c for c in cols if c in table.columns]].to_string(), flush=True)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
