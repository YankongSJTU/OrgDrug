# -*- coding: utf-8 -*-
"""
Organoid LOOCV Baselines — LightGBM + deep model transfers
=========================================
Fills Table 2 of the OrgDrug manuscript: leave-one-organoid-out transfer
baselines for the CRC / PDAC / BLCA cohorts, under the same protocol as
OrgDrug (frozen encoders, fine-tune interaction layers + head, lr 5e-4,
<=100 epochs patience 20, batch 64; early-stop on the left-out organoid).

- LightGBM : organoid-only LOOCV on [scF emb | Morgan FP] (no transfer).
- DeepCDR / DrugCell : retrained on GDSC2 with the scF-embedding cell view
  (the only cross-platform-comparable view), then transferred per cohort.
- MOLI     : source model keeps multi-omics branches (scF+mut+CNA);
  mutation/CNA views are zeroed at organoid inference (missing views).
- DeepTTA  : scF "barcode" CNN + SMILES Transformer; SMILES merged from
  gdsc_drug_smiles.csv + pdac_drug_smiles.csv (+ any resolved extras).

Usage (from GDRNet/src, via run_py.sh):
  /export/home/kongyan/run_py.sh rerun_organoid_baselines.py --gpu 0
"""

import os
import argparse
import io
import json
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("ORGDRUG_ROOT",
                 Path(__file__).resolve().parents[1]))
PROC = ROOT / "data/processed"
GEO_PROC = PROC / "geo"
FEAT = PROC / "orgdrug"
EXT = ROOT / "data/external"
OUT_DIR = ROOT / "results/organoid_baselines"
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cohorts", type=str, default="crc,pdac,blca")
parser.add_argument("--models", type=str,
                    default="LightGBM,DeepCDR,DrugCell,MOLI,DeepTTA")
parser.add_argument("--src-epochs", type=int, default=300)
parser.add_argument("--ft-epochs", type=int, default=100)
parser.add_argument("--batch", type=int, default=1024)
parser.add_argument("--ft-batch", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--ft-lr", type=float, default=5e-4)
parser.add_argument("--patience", type=int, default=40)
parser.add_argument("--ft-patience", type=int, default=20)
parser.add_argument("--wd", type=float, default=0.01)
args = parser.parse_args()

OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
SEED = args.seed

FREEZE_SUBSTR = {
    "DeepCDR": ["drug_enc", "cell_enc"],
    "DrugCell": ["cell_enc", "drug_enc"],
    "MOLI": ["exp_branch", "mut_branch", "cna_branch", "drug_branch"],
    "DeepTTA": ["cell_enc", "drug_enc"],
}

COHORT_FILES = {
    "crc":  dict(meta="organoid_pair_meta.csv", emb="organoid_cell_emb.npy",
                 ids="organoid_cell_ids.npy", fp="organoid_drug_features.npy",
                 y="organoid_response.npy", geodir=PROC),
    "pdac": dict(meta="pdac_pair_meta.csv", emb="pdac_cell_emb.npy",
                 ids="pdac_cell_ids.npy", fp="pdac_drug_features.npy",
                 y="pdac_response.npy", geodir=GEO_PROC),
    "blca": dict(meta="blca_pair_meta.csv", emb="blca_cell_emb.npy",
                 ids="blca_cell_ids.npy", fp="blca_drug_features.npy",
                 y="blca_response.npy", geodir=GEO_PROC),
}


def compute_metrics(y, p, name=""):
    from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
    if len(np.unique(y)) < 2:
        return dict(pearson=np.nan, r2=np.nan, rmse=np.nan, auroc=np.nan)
    r = float(np.corrcoef(y, p)[0, 1])
    r2 = float(r2_score(y, p))
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    thr = np.percentile(y, 30)
    try:
        auroc = float(roc_auc_score((y <= thr).astype(int), -p))
    except Exception:  # noqa: BLE001
        auroc = float("nan")
    if name:
        print(f"    {name}: r={r:.4f} R2={r2:.4f} RMSE={rmse:.4f} AUROC={auroc:.4f}",
              flush=True)
    return dict(pearson=r, r2=r2, rmse=rmse, auroc=auroc)


class PairDS(Dataset):
    """Generic tuple dataset."""
    def __init__(self, tensors):
        self.tensors = [torch.as_tensor(t) for t in tensors]

    def __len__(self):
        return len(self.tensors[-1])

    def __getitem__(self, i):
        return tuple(t[i] for t in self.tensors)


def load_cohort(cancer):
    cfg = COHORT_FILES[cancer]
    d = cfg["geodir"]
    meta = pd.read_csv(d / cfg["meta"])
    emb_all = np.load(d / cfg["emb"])
    ids_all = np.load(d / cfg["ids"], allow_pickle=True)
    fp_all = np.load(d / cfg["fp"]).astype(np.float32)
    y = np.load(d / cfg["y"]).astype(np.float32)
    m = {str(c): i for i, c in enumerate(ids_all)}
    scf = np.stack([emb_all[m[str(o)]] for o in meta["organoid_id"]]).astype(np.float32)
    fp = fp_all[:, :2048]
    assert len(y) == len(meta) == len(scf) == len(fp), \
        f"{cancer}: length mismatch {len(y)}/{len(meta)}/{len(scf)}/{len(fp)}"
    return meta, scf, fp, y


def smiles_ids_for(drug_names, vocab, pad_ids):
    from models.deeptta import encode_smiles
    smi = {}
    for csv in sorted(EXT.glob("*smiles*.csv")):
        try:
            sdf = pd.read_csv(csv)
            cnames = [c for c in sdf.columns if c.lower() in ("drug_name", "drug", "name")]
            snames = [c for c in sdf.columns if c.lower() in ("smiles", "canonical_smiles")]
            if cnames and snames:
                smi.update(dict(zip(sdf[cnames[0]].astype(str),
                                    sdf[snames[0]].astype(str))))
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] skip {csv.name}: {e}", flush=True)
    out, hit = [], 0
    for d in map(str, drug_names):
        if d in smi and smi[d] and smi[d].lower() != "nan":
            out.append(encode_smiles(smi[d], vocab))
            hit += 1
        else:
            out.append(pad_ids)
    print(f"    SMILES coverage: {hit}/{len(drug_names)} drugs", flush=True)
    return np.stack(out)


def train_source(model_name, data):
    """Train a source model on the full GDSC2 LCLO corpus (scF cell view)."""
    ckpt = OUT_DIR / f"src_{model_name}.pt"
    n, d_scf, d_fp = len(data.y), 3072, 2048

    if model_name == "MOLI":
        from models.moli import MOLI, MOLIDataset, train_moli
        omics = {}
        for name, mat_file, id_file in [("mut", "mut_matrix.npy", "mut_cell_ids.npy"),
                                        ("cna", "cna_matrix.npy", "cna_cell_ids.npy")]:
            mat = np.load(FEAT / mat_file)
            ids = np.load(FEAT / id_file, allow_pickle=True)
            row = {str(c): i for i, c in enumerate(ids)}
            omics[name] = np.zeros((n, mat.shape[1]), dtype=np.float32)
            for j, c in enumerate(map(str, data.cell_ids)):
                if c in row:
                    omics[name][j] = mat[row[c]]
        model = MOLI(n_genes=d_scf, n_mut=omics["mut"].shape[1],
                     n_cna=omics["cna"].shape[1], fp_bits=d_fp)
        dims = dict(mut_dim=omics["mut"].shape[1], cna_dim=omics["cna"].shape[1])
        ds = MOLIDataset(data.X_scf, omics["mut"], omics["cna"], data.X_fp, data.y)
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            return model, dims
        model, _ = train_moli(model, ds, ds, n_epochs=args.src_epochs,
                              batch_size=args.batch, lr=args.lr,
                              patience=args.patience, device=DEVICE)
        torch.save(model.state_dict(), ckpt)
        return model, dims

    if model_name in ("DeepCDR", "DrugCell"):
        if model_name == "DeepCDR":
            from models.deepcdr import DeepCDR, DeepCDRDataset, train_deepcdr
            model = DeepCDR(n_genes=d_scf, fp_bits=d_fp)
        else:
            from models.drugcell import DrugCellSimplified, DrugCellDataset, train_drugcell
            model = DrugCellSimplified(n_genes=d_scf, fp_bits=d_fp)
            DeepCDRDataset = DrugCellDataset      # same (x, fp, y) layout
            train_deepcdr = train_drugcell
        ds = DeepCDRDataset(data.X_scf, data.X_fp, data.y)
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            return model, None
        model, _ = train_deepcdr(model, ds, ds, n_epochs=args.src_epochs,
                                 batch_size=args.batch, lr=args.lr,
                                 patience=args.patience, weight_decay=args.wd,
                                 device=DEVICE)
        torch.save(model.state_dict(), ckpt)
        return model, None

    if model_name == "DeepTTA":
        from models.deeptta import DeepTTA, DeepTTADataset, train_deeptta, build_vocab, encode_smiles
        sdf = pd.read_csv(EXT / "gdsc_drug_smiles.csv")
        smi_by_drug = dict(zip(sdf["drug_name"].astype(str), sdf["smiles"].astype(str)))
        vocab = build_vocab([s for s in smi_by_drug.values() if s.lower() != "nan"])
        pad = np.zeros(120, dtype=np.int64)
        X_smi = np.stack([encode_smiles(smi_by_drug.get(str(d), ""), vocab)
                          if str(d) in smi_by_drug else pad for d in data.drug_names])
        model = DeepTTA(n_genes=d_scf, vocab_size=len(vocab))
        ds = DeepTTADataset(data.X_scf, X_smi, data.y)
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            return model, dict(vocab=vocab, pad=pad)
        model, _ = train_deeptta(model, ds, ds, n_epochs=args.src_epochs,
                                 batch_size=args.batch, lr=args.lr,
                                 patience=args.patience, device=DEVICE)
        torch.save(model.state_dict(), ckpt)
        return model, dict(vocab=vocab, pad=pad)

    raise ValueError(model_name)


def forward_factory(model_name):
    if model_name in ("DeepCDR", "DrugCell"):
        return lambda m, b: m(b[0].to(DEVICE), b[1].to(DEVICE))
    if model_name == "MOLI":
        return lambda m, b: m(b[0].to(DEVICE), b[1].to(DEVICE),
                              b[2].to(DEVICE), b[3].to(DEVICE))
    if model_name == "DeepTTA":
        return lambda m, b: m(b[0].to(DEVICE), b[1].to(DEVICE))
    raise ValueError(model_name)


def finetune_loocv(model_name, src_model, cohort_tensors, extra):
    """Frozen-encoder LOOCV transfer; returns pooled predictions."""
    meta, scf, fp, y = cohort_tensors
    organoids = sorted(meta["organoid_id"].unique())
    n = len(y)
    preds_all = np.zeros(n, dtype=np.float32)

    fwd = forward_factory(model_name)
    freeze = FREEZE_SUBSTR[model_name]

    if model_name == "MOLI":
        zero_mut = np.zeros((n, extra["mut_dim"]), dtype=np.float32)
        zero_cna = np.zeros((n, extra["cna_dim"]), dtype=np.float32)
    else:
        zero_mut = zero_cna = None

    for fi, test_oid in enumerate(organoids):
        tr = (meta["organoid_id"] != test_oid).values
        te = ~tr

        # rebuild fresh source-initialized model per fold
        model = rebuild_like(src_model)
        model.load_state_dict(src_model.state_dict())
        for name, p in model.named_parameters():
            if any(s in name for s in freeze):
                p.requires_grad = False
        model = model.to(DEVICE)

        if model_name == "MOLI":
            tr_ds = PairDS((scf[tr], zero_mut[tr], zero_cna[tr], fp[tr], y[tr]))
            te_ds = PairDS((scf[te], zero_mut[te], zero_cna[te], fp[te], y[te]))
        elif model_name == "DeepTTA":
            X_smi = extra["X_smi"]
            tr_ds = PairDS((scf[tr], X_smi[tr], y[tr]))
            te_ds = PairDS((scf[te], X_smi[te], y[te]))
        else:
            tr_ds = PairDS((scf[tr], fp[tr], y[tr]))
            te_ds = PairDS((scf[te], fp[te], y[te]))

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=args.ft_lr, weight_decay=0.05)
        loader = DataLoader(tr_ds, batch_size=args.ft_batch, shuffle=True,
                            num_workers=0, pin_memory=True)

        best_rmse, best_state, wait = float("inf"), None, 0
        for _ in range(args.ft_epochs):
            model.train()
            # keep frozen encoder modules in eval so BatchNorm running stats
            # (learned on GDSC) are not corrupted by organoid inputs
            for mname, mod in model.named_modules():
                if mname and any(s in mname for s in freeze):
                    mod.eval()
            for b in loader:
                opt.zero_grad()
                loss = F.huber_loss(fwd(model, b), b[-1].to(DEVICE), delta=1.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                p_te = np.concatenate([
                    fwd(model, [t[j:j + 512] for t in te_ds.tensors]).cpu().numpy()
                    for j in range(0, len(te_ds), 512)])
            rmse = float(np.sqrt(np.mean((y[te] - p_te) ** 2)))
            if rmse < best_rmse - 1e-5:
                best_rmse, wait = rmse, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= args.ft_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
            model = model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            p_te = np.concatenate([
                fwd(model, [t[j:j + 512] for t in te_ds.tensors]).cpu().numpy()
                for j in range(0, len(te_ds), 512)])
        preds_all[te] = p_te
        if (fi + 1) % 10 == 0 or fi == len(organoids) - 1:
            print(f"    fold {fi + 1}/{len(organoids)} done", flush=True)

    return preds_all


def rebuild_like(src_model):
    """Re-instantiate an untrained copy of the same architecture."""
    import copy
    model = copy.deepcopy(src_model)
    for name, p in model.named_parameters():
        p.requires_grad_(True)
    return model


def lightgbm_loocv(cohort_tensors):
    meta, scf, fp, y = cohort_tensors
    organoids = sorted(meta["organoid_id"].unique())
    X = np.concatenate([scf, fp], axis=1)
    preds = np.zeros(len(y), dtype=np.float32)
    for test_oid in organoids:
        te = (meta["organoid_id"] == test_oid).values
        tr = ~te
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=63,
                              max_depth=10, subsample=0.8, colsample_bytree=0.6,
                              min_child_samples=10, reg_alpha=0.5, reg_lambda=0.5,
                              n_jobs=-1, random_state=SEED, verbose=-1)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te]).astype(np.float32)
    return preds


def main():
    from data_loader import load_gdsc_lclo

    t0 = time.time()
    models = [m.strip() for m in args.models.split(",")]
    cohorts = [c.strip() for c in args.cohorts.split(",")]

    src_models, extras = {}, {}
    deep_models = [m for m in models if m != "LightGBM"]
    if deep_models:
        print("Loading GDSC LCLO corpus for source training ...", flush=True)
        data = load_gdsc_lclo(n_splits=5, seed=SEED)
        for mname in deep_models:
            print(f"Training source model: {mname}", flush=True)
            ts = time.time()
            src_models[mname], ex = train_source(mname, data)
            extras[mname] = ex or {}
            print(f"  source {mname} done ({(time.time() - ts) / 60:.1f} min)", flush=True)

    summary = {}
    for cancer in cohorts:
        print(f"\n{'=' * 66}\n  Cohort: {cancer.upper()}\n{'=' * 66}", flush=True)
        tensors = load_cohort(cancer)
        meta, scf, fp, y = tensors
        print(f"  {len(meta)} pairs, {meta['organoid_id'].nunique()} organoids, "
              f"{meta['drug_name'].nunique()} drugs", flush=True)

        for mname in models:
            print(f"\n  [{mname}] {cancer}", flush=True)
            if mname == "LightGBM":
                preds = lightgbm_loocv(tensors)
            else:
                if mname == "DeepTTA":
                    from models.deeptta import encode_smiles
                    vocab = extras[mname]["vocab"]
                    pad = extras[mname]["pad"]
                    extras[mname]["X_smi"] = smiles_ids_for(meta["drug_name"], vocab, pad)
                preds = finetune_loocv(mname, src_models[mname], tensors, extras[mname])
            metrics = compute_metrics(y, preds, f"{mname}-{cancer}")
            summary.setdefault(mname, {})[cancer] = metrics
            pd.DataFrame({"organoid_id": meta["organoid_id"], "drug_name": meta["drug_name"],
                          "y": y, "pred": preds}).to_csv(
                OUT_DIR / f"preds_{mname}_{cancer}.csv", index=False)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    rows = []
    for mname, cs in summary.items():
        rows.append({"Model": mname,
                     **{f"{k}_pearson": round(v["pearson"], 4) for k, v in cs.items()},
                     **{f"{k}_auroc": round(v["auroc"], 4) for k, v in cs.items()}})
    pd.DataFrame(rows).to_csv(OUT_DIR / "table2_organoid_baselines.csv", index=False)
    print("\nSaved:", OUT_DIR / "table2_organoid_baselines.csv", flush=True)
    print(pd.DataFrame(rows).to_string(), flush=True)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
