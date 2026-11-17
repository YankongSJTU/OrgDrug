#!/usr/bin/env bash
# One-click demo: synthetic data -> 1-fold LCLO training -> metrics (CPU ok)
set -e
cd "$(dirname "$0")"
python demo/run_demo.py "$@"
