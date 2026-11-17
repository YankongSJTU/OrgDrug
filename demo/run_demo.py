# -*- coding: utf-8 -*-
"""One-click OrgDrug demo: synthetic data -> LCLO training -> metrics.

Runs the real training entry (src/train_orgdrug_lclo.py, variant `full`,
1 seed x 1 fold x 15 epochs) on the synthetic demo dataset, then prints the
fold metrics and the mutation-stratified summary. CPU takes ~1-2 min; CUDA
about 10 s.

Usage:  python demo/run_demo.py [--device cuda:0|cpu] [--epochs 15]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
import make_demo_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--folds", default="0")
    args = ap.parse_args()

    device = args.device
    if device is None:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 66)
    print("  OrgDrug demo — synthetic 80x30 LCLO smoke run")
    print("=" * 66, flush=True)

    # 1. demo data (generated on first run; seconds)
    if not (make_demo_data.PROC / "gdsc_metadata.parquet").exists():
        make_demo_data.build()
        print()

    # 2. real training entry, small budget
    cmd = [sys.executable, str(ROOT / "src" / "train_orgdrug_lclo.py"),
           "--variant", "full", "--seeds", "42", "--folds", args.folds,
           "--epochs", str(args.epochs), "--batch", "256",
           "--patience", "30", "--workers", "0", "--log-every", "5",
           "--device", device, "--out", "results/demo_lclo"]
    print("  $", " ".join(cmd), "\n", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit("demo training failed")

    # 3. show the numbers
    print("\n" + "=" * 66)
    ev = json.loads((ROOT / "results/demo_lclo/fold_evals.json").read_text())
    for fold, e in ev.items():
        print(f"  fold {fold}: pooled_r={e['pooled_pearson']:.4f}  "
              f"per_drug_r={e['per_drug_mean']:.4f}  "
              f"AUROC30={e['auroc_30']}")
    summ = ROOT / "results/demo_lclo/summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        mut = s.get("mutation_stratified", {})
        for g, m in mut.items():
            print(f"  {g}: MAE mut {m['mean_mae_mut']:.4f} vs wt {m['mean_mae_wt']:.4f}")
    print("\n  demo OK. next: README.md -> 'Full pipeline' for real data.")


if __name__ == "__main__":
    main()
