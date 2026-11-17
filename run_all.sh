#!/usr/bin/env bash
# OrgDrug full pipeline on real data (server-oriented).
#
# Prerequisites — see README.md "Data requirements" for exact file names:
#   data/raw/ccle/           DepMap OmicsExpression_TPM.csv, OmicsSomaticMutations_damaging.csv
#   data/raw/gdsc/           GDSC1/GDSC2_fitted_dose_response.csv
#   data/external/           gdsc_drug_smiles.csv
#   data/processed/          gdsc_* parquet + scfoundation_cell_* (GDRNet v1 ecosystem)
#   data/processed/geo/      organoid cohorts (CRC/PDAC/BLCA) for the transfer leg
#
# Env:
#   DEVICE   cuda:0 (default cuda:0)
#   SEEDS    42,123,456
#   PY       python interpreter (default python3)
#
# Usage:  ./run_all.sh [stage]     stage in:
#   features | lclo | ablation | pretrain | transfer | eval95 |
#   prism | transfer-prism | tox | all
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
DEVICE=${DEVICE:-cuda:0}
SEEDS=${SEEDS:-42,123,456}
STAGE=${1:-all}

features() {
  $PY src/prepare_features.py
}

lclo() {           # E1 head-to-head benchmark (5 folds x 3 seeds)
  $PY src/train_orgdrug_lclo.py --variant full --device "$DEVICE" --seeds "$SEEDS"
}

ablation() {       # E2 component ablations
  for V in no-mut no-path no-tgt no-tgca no-transformer no-cross no-list; do
    $PY src/train_orgdrug_lclo.py --variant $V --device "$DEVICE" --seeds 42
  done
}

pretrain() {       # E4 leg 1: expanded-set pretraining (classic shell)
  $PY src/pretrain_orgdrug_expanded.py --device "$DEVICE" --variant classic \
      --n-cross 3 --lambda-list 0.05 --view-dropout 0.05
}

transfer() {       # E4 leg 2: organoid LOOCV transfer (needs pretrain + v1 ckpts)
  $PY src/finetune_orgdrug_organoid.py --device "$DEVICE" \
      --cancers crc,pdac,blca --init-v1 --raw-protocol
}

eval95() {         # 95% decision-accuracy scoreboard
  $PY src/eval_accuracy95.py
}

prism() {          # E5: PRISM 19Q4 expansion prep + pretraining
  $PY src/prepare_prism.py
  $PY src/pretrain_orgdrug_expanded.py --device "$DEVICE" \
      --data-dir data/processed/expanded_prism --lambda-list 0.05
}

transfer-prism() { # E5b: PRISM-warm-start transfer + dual/mega ensembles
  $PY src/finetune_orgdrug_organoid.py --device "$DEVICE" --cancers pdac \
      --init-ckpt "orgdrug_prism_s{seed}.pt" --tag prism
  $PY src/ensemble_transfer.py
}

tox() {            # E7: PRISM-AUC toxicity endpoint
  $PY src/prepare_prism_tox.py
  $PY src/pretrain_orgdrug_expanded.py --device "$DEVICE" \
      --data-dir data/processed/prism_tox --dump-val
  $PY src/eval_tox95.py
}

case $STAGE in
  features)        features ;;
  lclo)            lclo ;;
  ablation)        ablation ;;
  pretrain)        pretrain ;;
  transfer)        transfer ;;
  eval95)          eval95 ;;
  prism)           prism ;;
  transfer-prism)  transfer-prism ;;
  tox)             tox ;;
  all)             features; lclo; ablation; pretrain; transfer; eval95 ;;
  *) echo "unknown stage: $STAGE (see header of run_all.sh)"; exit 1 ;;
esac
