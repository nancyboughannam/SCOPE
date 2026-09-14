#!/usr/bin/env bash
# Run from the directory containing the original models and candidate dataset.
set -euo pipefail
# Locate FLaskAPIs from this script's location.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$SCRIPT_DIR")"

export SCOPE_RESULTS_DIR="$PROJECT_DIR/scope_candidate_xgboost_results"
export SCOPE_CANDIDATE_DATASET="$PROJECT_DIR/scope_candidate_dataset/candidate_dataset_train.csv"
if [[ $# -ne 3 ]]; then
  echo 'Usage: bash start_catalogue_service.sh FRACTION SUBSET_SEED NEW_DECISION_ROOT'
  echo 'Example: bash start_catalogue_service.sh 0.25 1 catalogue_runs/pilot_25_s1'
  exit 2
fi
if [[ -e "$3" ]]; then
  echo 'Choose a new decision root: the supplied path already exists.'
  exit 2
fi
export SCOPE_CATALOGUE_FRACTION="$1"
export SCOPE_CATALOGUE_SEED="$2"
export SCOPE_ABLATION_RESULTS_DIR="$3"
export SCOPE_LAMBDA=0.20
export SCOPE_R_MAX=0.15
export SCOPE_W_FAILURE=0.50
export SCOPE_W_SERVICE=0.50
export SCOPE_HARD_STABILITY=1
export SCOPE_PORT=5002
unset SCOPE_DECISION_LOG
exec python3 "$SCRIPT_DIR/ml_optimizer_service_catalogue.py"