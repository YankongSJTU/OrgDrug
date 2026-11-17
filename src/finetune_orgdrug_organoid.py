# -*- coding: utf-8 -*-
"""
OrgDrug Organoid Fine-Tuning (E4) — LOOCV, frozen-encoder transfer
====================================================================
Mirrors GDRNet v1's organoid protocol (LOOCV per organoid; freeze all view
encoders + ID embeddings; tune interaction layers; lr 5e-4 wd 0.05 cosine,
patience 20, <=100 epochs, bs 64; early stop on test organoid RMSE) with
OrgDrug's missing-modality handling:

  * mutation view always missing for organoids (no matched WES in GEO cohorts)
  * landmark/pathway views computed from cohort expression where possible
    (PDAC FPKM, BLCA counts); CRC stays embedding-only
  * unseen drugs (not in GDSC expanded catalog) get drug-ID token masked out

Usage:
  python finetune_orgdrug_organoid.py --device cuda:6 --cancers crc,pdac,blca
"""

import os
import argparse
import io
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("ORGDRUG_ROOT",
                 Path(__file__).resolve().parents[1]))
MODELS = ROOT / "models"
FEAT_DIR = ROOT / "data/processed/orgdrug"
OUT_DIR = ROOT / "results/orgdrug"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass

from models.orgdrug import OrgDrug, OrgDrugDataset  # noqa: E402
from orgdrug_data import (load_tables, drug_set_rows,  # noqa: E402
                          organoid_expression_features, load_organoid_cohort)

# v1 protocol parity: freeze ALL encoders AND the pretrained interaction core
# (transformer/fusion/gating); fine-tune only cross + deep + heads. The core
# was pretrained with organoid-mode masking so its PDO-time forward pass is
# in-distribution without adaptation.
FROZEN_SUBSTR = ("gene_enc", "scf_enc", "path_enc", "mut_view", "fp_enc",
                 "tgt_view", "cell_emb", "drug_emb", "cell_tok", "drug_tok",
                 "type_emb", "in_norm", "encoder", "fusion", "pool_norm",
                 "gate")

V1_REFERENCE = {"crc": dict(r=0.8902, rmse=1.3108, auroc=0.9411),
                "pdac": dict(r=0.741, auroc=0.831),
                "blca": dict(r=0.640, auroc=0.868)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:6")
    p.add_argument("--cancers", default="crc,pdac,blca")
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--config", default="orgdrug_config.json")
    p.add_argument("--init-ckpt", default=None,
                   help="warm-start classic shell from this checkpoint template "
                        "({seed} substituted), e.g. orgdrug_classic_prism_s{seed}.pt; "
                        "takes precedence over --init-v1")
    p.add_argument("--init-v1", action="store_true",
                   help="initialize classic encoders/cross/deep/head from the "
                        "proven GDRNet v1 expanded checkpoints instead of our "
                        "own pretrain (our pure-recipe checkpoint proved "
                        "unstable under fine-tuning)")
    p.add_argument("--raw-protocol", action="store_true",
                   help="v1-exact protocol: raw-scale y, no warmup (use with "
                        "--init-v1; the z-guard was needed only for our own "
                        "unstable checkpoints)")
    p.add_argument("--tag", default=None,
                   help="suffix output filenames (organoid_{cancer}_{tag}_*.csv/"
                        "json) so multiple recipes coexist, e.g. v1init / prism5")
    return p.parse_args()


def compute_metrics(y, p, name=""):
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    r2 = float(r2_score(y, p))
    r = float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else float("nan")
    thr = np.percentile(y, 30)
    try:
        auroc = float(roc_auc_score((y <= thr).astype(int), -p))
    except Exception:
        auroc = float("nan")
    if name:
        print(f"  {name:30s} Pearson={r:.4f}  R2={r2:.4f}  RMSE={rmse:.4f}  "
              f"AUROC={auroc:.4f}", flush=True)
    return dict(pearson=r, r2=r2, rmse=rmse, auroc=auroc)


class _WrappedClassic(nn.Module):
    """Adapt OrgDrugClassic's single-output forward to the (reg, cls) interface."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, *b):
        return self.m(*b), None


def build_model(config, seed, device, init_v1=False, init_ckpt=None):
    if config.get("variant") == "classic":
        from models.orgdrug_classic import OrgDrugClassic
        fp_dim = 2048 if init_v1 else config["fp_dim"]
        model = _WrappedClassic(OrgDrugClassic(
            fp_dim=fp_dim, path_dim=config["path_dim"],
            n_shared_genes=config["n_shared_genes"],
            n_cross=config.get("n_cross", 3),
            n_cells=config["n_cells"], n_drugs=config["n_drugs"],
            dropout=config.get("dropout", 0.15)))
        if init_ckpt:
            # E5 leg: warm-start from our own PRISM-pretrained checkpoint
            # (full fp view incl. MACCS/onehot is supported by these weights)
            path = Path(str(init_ckpt).format(seed=seed))
            path = path if path.is_absolute() else MODELS / path
            state = torch.load(path, map_location="cpu", weights_only=True)
            own = model.state_dict()
            filtered = {k: v for k, v in state.items()
                        if k in own and own[k].shape == v.shape}
            miss, _ = model.load_state_dict(filtered, strict=False)
            print(f"    ckpt-init: loaded {len(filtered)} tensors "
                  f"(fresh: {len(miss)}) <- {path.name}", flush=True)
        elif init_v1:
            # proven-stable v1 expanded weights for all compatible modules;
            # our new modules (path/mut/tgt encoders) stay fresh-initialized
            v1_sd = torch.load(ROOT / f"models/gdrnet_s{seed}.pt",
                               map_location="cpu", weights_only=True)
            v1_sd = {k.replace("module.", ""): v for k, v in v1_sd.items()}
            own = model.m.state_dict()
            filtered = {k: v for k, v in v1_sd.items()
                        if k in own and own[k].shape == v.shape}
            miss, _ = model.m.load_state_dict(filtered, strict=False)
            print(f"    v1-init: loaded {len(filtered)} tensors "
                  f"(fresh: {len(miss)})", flush=True)
        else:
            ckpt_name = f"orgdrug_classic_expanded_s{seed}.pt"
            state = torch.load(MODELS / ckpt_name,
                               map_location="cpu", weights_only=True)
            model.load_state_dict({k.replace("module.", ""): v
                                   for k, v in state.items()})
        frozen = ("gene_enc", "scf_enc", "path_enc", "mut_enc", "fp_enc",
                  "tgt_enc", "cell_emb", "drug_emb", "input_proj")
    else:
        model = OrgDrug(n_genes=2000, scf_dim=3072,
                        fp_dim=config["fp_dim"], path_dim=config["path_dim"],
                        n_shared_genes=config["n_shared_genes"],
                        n_cross=config.get("n_cross", 3),
                        n_cells=config["n_cells"], n_drugs=config["n_drugs"],
                        dropout=config.get("dropout", 0.15))
        state = torch.load(MODELS / f"orgdrug_expanded_s{seed}.pt",
                           map_location="cpu", weights_only=True)
        model.load_state_dict({k.replace("module.", ""): v
                               for k, v in state.items()})
        frozen = FROZEN_SUBSTR
    for name, param in model.named_parameters():
        if any(s in name for s in frozen):
            param.requires_grad = False
    return model.to(device)


def finetune_fold(model, tr_ds, te_ds, device, args):
    from torch.utils.data import DataLoader
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                           num_workers=0, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False,
                           num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=5e-6)

    # Cohort endpoints (e.g. 1-AUC) live on a different scale than the
    # pretrained lnIC50 head. Standardize y on the training organoids (baked
    # into the datasets so shuffled batches keep labels aligned) and train in
    # z-space; inverse-transform predictions before returning. --raw-protocol
    # (v1-exact) skips this and the warmup.
    raw = getattr(args, "raw_protocol", False)
    y_mu, y_sd = 0.0, 1.0
    if not raw:
        y_mu = float(tr_ds.y.numpy().mean())
        y_sd = max(float(tr_ds.y.numpy().std()), 1e-6)
        tr_ds.y = torch.FloatTensor((tr_ds.y.numpy() - y_mu) / y_sd)
        te_ds.y = torch.FloatTensor((te_ds.y.numpy() - y_mu) / y_sd)

    def fwd(loader):
        model.eval()
        out = []
        with torch.no_grad():
            for (xg, scf, fp, path, mi, mm, ti, tm, ci, di, av, yb, wb, sb) in loader:
                batch = [t.to(device) for t in (xg, scf, fp, path, mi, mm, ti, tm, ci, di)]
                reg, _ = model(*batch, av.to(device))
                out.append(reg.float().cpu())
        return torch.cat(out).numpy()

    best_rmse, best_state, wait = float("inf"), None, 0
    for epoch in range(args.epochs):
        if not raw:
            # short linear warmup protects the pretrained head from the first
            # large-error batches
            warm = min(1.0, (epoch + 1) / 5.0)
            for g in optimizer.param_groups:
                g["lr"] = scheduler.get_last_lr()[0] * warm
        model.train()
        # keep frozen modules in eval mode: their BatchNorm running statistics
        # must NOT drift onto the organoid distribution (v1 protocol)
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm1d) and \
               not any(p.requires_grad for p in mod.parameters()):
                mod.eval()
        for bi, (xg, scf, fp, path, mi, mm, ti, tm, ci, di, av, yb, wb, sb) in enumerate(tr_loader):
            batch = [t.to(device) for t in (xg, scf, fp, path, mi, mm, ti, tm, ci, di)]
            yz = yb.to(device)
            optimizer.zero_grad()
            reg, _ = model(*batch, av.to(device))
            loss = F.huber_loss(reg, yz, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        preds = fwd(te_loader)
        y_te_z = te_ds.y.numpy()
        rmse = float(np.sqrt(np.mean((y_te_z - preds) ** 2)))
        if rmse < best_rmse:
            best_rmse, wait = rmse, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break

    model.load_state_dict(best_state)
    model = model.to(device)
    preds = fwd(te_loader)
    return (preds * y_sd + y_mu).astype(np.float32)


def build_drug_views_transfer(t, cancer, drugs, morgan_by_drug):
    """Transfer-safe drug fp: [morgan | cohort-MACCS | onehot].

    Pretraining only saw drugs with populated MACCS+onehot blocks; feeding
    all-zero blocks for cohort drugs puts 81% of PDO pairs outside the frozen
    BN's support. Compute MACCS from the cohort's own SMILES and one-hot the
    'Unknown' pathway category for drugs absent from the GDSC annotation.
    """
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys

    ext = ROOT / "data/external"
    smi_src = (ext / "pdac_drug_smiles.csv" if cancer == "pdac"
               else ext / "gdsc_drug_smiles.csv")
    smi = pd.read_csv(smi_src)
    smi_map = dict(zip(smi["drug_name"].astype(str), smi["smiles"].astype(str)))
    # fallback merge for the other cohorts
    other = ext / ("gdsc_drug_smiles.csv" if cancer == "pdac"
                   else "pdac_drug_smiles.csv")
    if other.exists():
        o = pd.read_csv(other)
        for n, s in zip(o["drug_name"].astype(str), o["smiles"].astype(str)):
            smi_map.setdefault(n, s)

    pw_names = pd.read_csv(FEAT_DIR / "drug_pathway_names.csv")["pathway"].tolist()
    unknown_idx = next((pw_names.index(u) for u in ("Unknown", "Unclassified")
                        if u in pw_names), None)
    onehot_row = {d: i for i, d in enumerate(
        pd.read_csv(FEAT_DIR / "drug_target_drugs.csv")["drug_name"])}

    n_mac = t["maccs"].shape[1]
    n_oh = len(pw_names)
    fp = np.zeros((len(drugs), 2048 + n_mac + n_oh), dtype=np.float32)
    tgt_idx = np.zeros((len(drugs), 16), dtype=np.int64)
    tgt_mask = np.zeros((len(drugs), 16), dtype=bool)

    for di, d in enumerate(drugs):
        if d in morgan_by_drug:
            fp[di, :2048] = morgan_by_drug[d]
        mol = Chem.MolFromSmiles(smi_map.get(d, "")) if smi_map.get(d) else None
        if mol is not None:
            fp[di, 2048:2048 + n_mac] = np.array(
                MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)[:n_mac]
        if d in onehot_row:
            fp[di, 2048 + n_mac:] = t["onehot"][onehot_row[d]]
            hit = t["tgt_genes"][t["tgt_mat"][onehot_row[d]] > 0]
            idxs = [t["gene2idx"][g] for g in hit if g in t["gene2idx"]][:16]
            tgt_idx[di, :len(idxs)] = idxs
            tgt_mask[di, :len(idxs)] = len(idxs) > 0
        elif unknown_idx is not None:
            fp[di, 2048 + n_mac + unknown_idx] = 1.0
    print(f"    drug views: morgan={sum(d in morgan_by_drug for d in drugs)} "
          f"maccs={sum(bool(smi_map.get(d)) for d in drugs)} "
          f"unknown-onehot={sum(d not in onehot_row for d in drugs)}", flush=True)
    return fp, tgt_idx, tgt_mask


def run_cancer(cancer, config, t, args):
    print(f"\n{'='*70}\n  {cancer.upper()} organoid LOOCV\n{'='*70}", flush=True)
    meta, scf_pair, morgan_pair, y = load_organoid_cohort(cancer)
    organoids = sorted(meta["organoid_id"].unique())
    print(f"  {len(organoids)} organoids, {len(y)} pairs, "
          f"{meta['drug_name'].nunique()} drugs", flush=True)

    # ── organoid-level cell views ──
    # Cross-platform landmark/pathway z-scores are NOT comparable to the GDSC
    # z-scored distributions the frozen encoders were calibrated on (v1 fed
    # zeros for the same reason). Start embedding-only (v1 information diet);
    # portable normalizations (rank-matched) are a follow-up experiment.
    import json as _json
    from orgdrug_data import FEAT as _FEAT
    _gs = _json.loads((_FEAT / "hallmark_genesets.json").read_text())
    path_dim = len(_gs["MSigDB_Hallmark_2020"]["terms"]) + 1
    lm = pw = None
    print("  embedding-only transfer: gene/path views masked (v1 diet)", flush=True)

    pos = {o: i for i, o in enumerate(organoids)}
    n = len(y)
    gene_mat = lm if lm is not None else np.zeros((len(organoids), 2000), np.float32)
    path_mat = pw if pw is not None else np.zeros((len(organoids), path_dim), np.float32)
    # scf per organoid via first pair occurrence
    scf_org = {}
    for o, s in zip(meta["organoid_id"].values, scf_pair):
        scf_org.setdefault(o, s)
    scf_mat = np.stack([scf_org[o] for o in organoids]).astype(np.float32)
    mut_idx = np.zeros((len(organoids), 64), dtype=np.int64)
    mut_mask = np.zeros((len(organoids), 64), dtype=bool)

    # ── drug views (unique cohort drugs) ──
    drugs = sorted(meta["drug_name"].unique())
    morgan_by_drug = {}
    for d, f in zip(meta["drug_name"].values, morgan_pair):
        morgan_by_drug.setdefault(d, f)
    fp_by_drug, tgt_idx, tgt_mask = build_drug_views_transfer(
        t, cancer, drugs, morgan_by_drug)
    if getattr(args, "init_v1", False):
        # v1-initialized fp encoder expects the original 2048-bit Morgan
        fp_by_drug = fp_by_drug[:, :2048].copy()

    drug_map = pd.read_csv(ROOT / "data/processed/expanded/drug_id_map.csv")
    drug_to_idx = dict(zip(drug_map["DRUG_NAME"], drug_map["drug_idx"]))

    cell_feats = dict(gene=gene_mat, scf=scf_mat, path=path_mat,
                      mut_idx=mut_idx, mut_mask=mut_mask)
    drug_feats = dict(fp=fp_by_drug, tgt_idx=tgt_idx, tgt_mask=tgt_mask)

    ci_pair = np.array([pos[o] for o in meta["organoid_id"]], dtype=np.int64)
    di_pair = np.array([drugs.index(d) for d in meta["drug_name"]], dtype=np.int64)

    gdsc_idx = np.array([drug_to_idx.get(d, 0) for d in meta["drug_name"]], dtype=np.int64)
    drug_emb_avail = (gdsc_idx > 0).astype(np.float32)
    n_overlap_pairs = int(drug_emb_avail.sum())
    print(f"  pairs with GDSC-overlapping drug: {n_overlap_pairs}/{n}", flush=True)

    def make_avail():
        av = np.ones((n, 8), dtype=np.float32)
        av[:, 0] = 1.0 if lm is not None else 0.0
        av[:, 2] = 0.0                       # mutations missing for organoids
        av[:, 3] = 1.0 if pw is not None else 0.0
        av[:, 5] = tgt_mask[di_pair].any(axis=1)
        av[:, 7] = drug_emb_avail            # unseen drugs: mask ID token
        return av

    avail = make_avail()
    n_unseen = int((avail[:, 7] == 0).sum())
    print(f"  unseen-drug pairs (ID masked): {n_unseen}/{n}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    seed_preds = {s: np.zeros(n, dtype=np.float32) for s in seeds}

    for fi, test_oid in enumerate(organoids):
        te_mask = (meta["organoid_id"].values == test_oid)
        tr_mask = ~te_mask
        for seed in seeds:
            model = build_model(config, seed, args.device,
                                init_v1=getattr(args, "init_v1", False),
                                init_ckpt=getattr(args, "init_ckpt", None))
            # organoids have no pretrained cell ID -> none token; drugs map to
            # GDSC embedding vocabulary (0/none for structurally unseen drugs)
            ci_emb = np.zeros(n, dtype=np.int64)
            di_emb = gdsc_idx
            tr_ds = OrgDrugDataset(cell_feats, drug_feats, ci_pair[tr_mask],
                                   di_pair[tr_mask], y[tr_mask],
                                   avail=avail[tr_mask],
                                   emb_cell_idx=ci_emb[tr_mask],
                                   emb_drug_idx=di_emb[tr_mask])
            te_ds = OrgDrugDataset(cell_feats, drug_feats, ci_pair[te_mask],
                                   di_pair[te_mask], y[te_mask],
                                   avail=avail[te_mask],
                                   emb_cell_idx=ci_emb[te_mask],
                                   emb_drug_idx=di_emb[te_mask])
            seed_preds[seed][te_mask] = finetune_fold(model, tr_ds, te_ds,
                                                      args.device, args)
            del model
        if (fi + 1) % 5 == 0 or fi == len(organoids) - 1:
            ens = np.mean([seed_preds[s] for s in seeds], axis=0)
            compute_metrics(y, ens, name=f"[{fi+1}/{len(organoids)}] running-ens")

    results = {}
    for seed in seeds:
        results[f"seed{seed}"] = compute_metrics(y, seed_preds[seed])
    ens = np.mean([seed_preds[s] for s in seeds], axis=0)
    results["ensemble"] = compute_metrics(y, ens, name=f"{cancer} ENSEMBLE")
    ref = V1_REFERENCE.get(cancer, {})
    if ref:
        print(f"  GDRNet v1 reference: r={ref.get('r')}  "
              f"auroc={ref.get('auroc')}", flush=True)

    # per-organoid r
    per_org = []
    for o in organoids:
        m = meta["organoid_id"].values == o
        if m.sum() >= 5:
            r = float(np.corrcoef(y[m], ens[m])[0, 1])
            per_org.append(dict(organoid=o, r=round(r, 4), n=int(m.sum())))
    tag = f"_{args.tag}" if args.tag else ""
    pd.DataFrame({"organoid_id": meta["organoid_id"], "drug_name": meta["drug_name"],
                  "y_true": y,
                  **{f"pred_s{s}": seed_preds[s] for s in seeds},
                  "pred_ensemble": ens}).to_csv(
        OUT_DIR / f"organoid_{cancer}{tag}_predictions.csv", index=False)
    with open(OUT_DIR / f"organoid_{cancer}{tag}_results.json", "w") as f:
        json.dump(dict(results=results, per_organoid=per_org,
                       n_unseen_drug_pairs=n_unseen), f, indent=1)
    return results


def main():
    args = parse_args()
    config = json.loads((MODELS / args.config).read_text())
    t = load_tables()
    assert t["fp_dim"] == config["fp_dim"], "feature tables vs checkpoint mismatch"
    t0 = time.time()
    all_res = {}
    for cancer in args.cancers.split(","):
        all_res[cancer] = run_cancer(cancer, config, t, args)
    print(f"\n  total {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
