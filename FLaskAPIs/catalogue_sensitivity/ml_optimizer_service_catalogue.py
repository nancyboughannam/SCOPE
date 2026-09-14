"""SCOPE candidate-aware joint optimizer service.

This service replaces the old CPU-load heuristic used by RT_OPT_ML. For every
optimization window it:

1. reconstructs the exact state features used during model training;
2. evaluates every feasible candidate from the training candidate catalogue;
3. predicts task-failure rate and deadline-normalized service time per app;
4. balances normalized predicted performance against normalized
   reconfiguration cost; and
5. returns the lowest-cost candidate in the legacy Java response format.

Expected directory layout (all paths may be overridden by environment vars):

FLaskAPIs/
  ml_optimizer_service.py
  scope_candidate_dataset/
    candidate_dataset_train.csv
  scope_candidate_xgboost_results/
    model_manifest.json
    models/
      xgb_app_0_failure_rate.json
      ... six model files in total ...

===== SCOPE JOINT OBJECTIVE CHANGE START =====

bash catalogue_sensitivity/start_catalogue_service.sh 0.25 1 catalogue_runs/pilot_quarter_s1/ite2/decisions
bash catalogue_sensitivity/start_catalogue_service.sh 0.5 1 catalogue_runs/pilot_half_s1/ite2/decisions
bash catalogue_sensitivity/start_catalogue_service.sh 1.0 1 catalogue_runs/pilot_full_s1/ite2/decisions


bash catalogue_sensitivity/start_catalogue_service.sh 0.25 2 catalogue_runs/pilot_quarter_s2/ite2/decisions
bash catalogue_sensitivity/start_catalogue_service.sh 0.5 2 catalogue_runs/pilot_half_s2/ite2/decisions
bash catalogue_sensitivity/start_catalogue_service.sh 1.0 2 catalogue_runs/pilot_full_s2/ite2/decisions


"""

from __future__ import annotations
import json
import csv
import logging
from catalogue_subset import select_catalogue
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, request


APP_COUNT = 3
PORT = int(os.getenv("SCOPE_PORT", "5002"))
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(
    os.getenv(
        "SCOPE_RESULTS_DIR",
        str(BASE_DIR / "scope_candidate_xgboost_results"),
    )
)
MANIFEST_PATH = RESULTS_DIR / "model_manifest.json"
CANDIDATE_DATASET_PATH = Path(
    os.getenv(
        "SCOPE_CANDIDATE_DATASET",
        str(BASE_DIR / "scope_candidate_dataset" / "candidate_dataset_train.csv"),
    )
)
# ===== ABLATION-SAFE DECISION LOGGING CHANGE START =====
# SCOPE_DECISION_LOG remains an optional exact-file override. When it is not
# supplied, the service creates one file per variant, iteration, seed, and
# vehicle count, preventing runs from being silently mixed or overwritten.
DECISION_LOG_OVERRIDE = os.getenv("SCOPE_DECISION_LOG", "").strip()
ABLATION_RESULTS_DIR = Path(
    os.getenv(
        "SCOPE_ABLATION_RESULTS_DIR",
        str(BASE_DIR / "scope_ablation_results"),
    )
)
# ===== ABLATION-SAFE DECISION LOGGING CHANGE END =====

# Joint-objective controls. Sweep SCOPE_LAMBDA in the final experiments.
LAMBDA = float(os.getenv("SCOPE_LAMBDA", "0.50"))
W_FAILURE = float(os.getenv("SCOPE_W_FAILURE", "0.50"))
W_SERVICE = float(os.getenv("SCOPE_W_SERVICE", "0.50"))

# ===== CHANGE START: HARD STABILITY CONFIGURATION =====

HARD_STABILITY_ENABLED = (
    os.getenv("SCOPE_HARD_STABILITY", "1").strip().lower()
    in {"1", "true", "yes", "on"}
)

R_MAX = float(os.getenv("SCOPE_R_MAX", "0.20"))

# ===== CHANGE END: HARD STABILITY CONFIGURATION =====


# Training-set target support. These constants normalize both performance
# targets to approximately [0, 1] without using validation/test information.
TRAIN_FAILURE_MIN = 0.0
TRAIN_FAILURE_MAX = 0.7353648757016841
TRAIN_DNST_MIN = 0.5327461065548065
TRAIN_DNST_MAX = 1.6538610651089858

# Feasible configuration support used during candidate-data collection.
ALPHA_MIN, ALPHA_MAX = 0.05, 0.95
W_LOAD_MIN, W_LOAD_MAX = 0.10, 0.90
THRESHOLD_MIN, THRESHOLD_MAX = 20.0, 95.0

CANDIDATE_COLUMNS = [
    "candidate_alpha",
    "candidate_w_load_0",
    "candidate_w_load_1",
    "candidate_w_load_2",
    "candidate_threshold_0",
    "candidate_threshold_1",
    "candidate_threshold_2",
]
TARGETS = ("failure_rate", "deadline_norm_service_time")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [SCOPE-JOINT] %(message)s",
)
logger = logging.getLogger(__name__)
app = Flask(__name__)
decision_log_lock = threading.Lock()


def _validate_objective_configuration() -> None:
    if not 0.0 <= LAMBDA <= 1.0:
        raise ValueError("SCOPE_LAMBDA must be between 0 and 1")
    if W_FAILURE < 0.0 or W_SERVICE < 0.0:
        raise ValueError("Performance weights cannot be negative")
    if not np.isclose(W_FAILURE + W_SERVICE, 1.0, atol=1e-9):
        raise ValueError("SCOPE_W_FAILURE + SCOPE_W_SERVICE must equal 1")

    # ===== CHANGE START: VALIDATE HARD STABILITY LIMIT =====

    if not 0.0 <= R_MAX <= 1.0:
        raise ValueError("SCOPE_R_MAX must be between 0 and 1")

    # ===== CHANGE END: VALIDATE HARD STABILITY LIMIT =====


# ===== EXPLICIT ABLATION MODES CHANGE START =====
def _resolve_objective_settings(data: dict[str, Any]) -> dict[str, Any]:
    """Map an explicit simulator policy to one controlled formulation."""
    policy = str(data.get("policy", "RT_OPT_ML")).strip().upper()

    if policy == "SCOPE":
        variant = "SCOPE"
        lambda_value = LAMBDA
        hard_enabled = True
    elif policy == "SOFTCOST_OPT":
        variant = "SOFTCOST_OPT"
        lambda_value = LAMBDA
        hard_enabled = False
    elif policy == "PERFORMANCE_ONLY":
        variant = "PERFORMANCE_ONLY"
        lambda_value = 0.0
        hard_enabled = False
    elif policy == "RT_OPT_ML":
        # Backward-compatible alias used by the main SCOPE comparison and
        # earlier lambda/R_max sensitivity experiments.
        variant = "SCOPE"
        lambda_value = LAMBDA
        hard_enabled = HARD_STABILITY_ENABLED
    else:
        raise ValueError(
            "Port 5002 accepts only SCOPE, SOFTCOST_OPT, "
            "PERFORMANCE_ONLY, or the legacy RT_OPT_ML alias; "
            f"received {policy!r}"
        )

    return {
        "policy": policy,
        "variant": variant,
        "lambda": float(lambda_value),
        "hard_stability_enabled": bool(hard_enabled),
        "r_max": float(R_MAX),
    }


def _safe_file_token(value: Any, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    token = token.strip("._-")
    return token or fallback


def _decision_log_path(data: dict[str, Any], variant: str) -> Path:
    if DECISION_LOG_OVERRIDE:
        return Path(DECISION_LOG_OVERRIDE)

    run_id = _safe_file_token(data.get("run_id", "run_unspecified"), "run_unspecified")
    iteration = _safe_file_token(data.get("iteration", "unknown"), "unknown")
    seed = _safe_file_token(data.get("simulation_seed", "unknown"), "unknown")
    devices = _safe_file_token(data.get("num_devices", "unknown"), "unknown")
    variant_token = _safe_file_token(variant.lower(), "unknown_variant")
    filename = (
        f"{variant_token}_iteration{iteration}_seed{seed}_n{devices}_"
        f"{run_id}.csv"
    )
    return ABLATION_RESULTS_DIR / "decision_logs" / variant_token / filename
# ===== EXPLICIT ABLATION MODES CHANGE END =====

def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing model manifest: {MANIFEST_PATH}. "
            "Place scope_candidate_xgboost_results beside this service."
        )
    import json

    with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_models(
    manifest: dict[str, Any],
) -> dict[tuple[int, str], dict[str, Any]]:
    loaded: dict[tuple[int, str], dict[str, Any]] = {}
    for item in manifest.get("models", []):
        app_id = int(item["app_id"])
        target = str(item["target"])
        if app_id not in range(APP_COUNT) or target not in TARGETS:
            continue

        model_path = RESULTS_DIR / item["model_file"]
        if not model_path.exists():
            raise FileNotFoundError(f"Missing XGBoost model: {model_path}")

        # Native Booster loading avoids the scikit-learn _estimator_type
        # compatibility problem that occurred while saving the models.
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        loaded[(app_id, target)] = {
            "booster": booster,
            "features": list(item["features"]),
            "best_iteration": int(item["best_iteration"]),
            "path": model_path,
        }

    expected = {(app_id, target) for app_id in range(APP_COUNT) for target in TARGETS}
    missing = expected - set(loaded)
    if missing:
        raise RuntimeError(f"The manifest is missing models: {sorted(missing)}")
    return loaded


def _load_candidate_catalogue() -> pd.DataFrame:
    if not CANDIDATE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing candidate training split: {CANDIDATE_DATASET_PATH}. "
            "Place scope_candidate_dataset beside this service or set "
            "SCOPE_CANDIDATE_DATASET."
        )

    catalogue = pd.read_csv(
        CANDIDATE_DATASET_PATH,
        usecols=CANDIDATE_COLUMNS,
    )
    catalogue = catalogue.apply(pd.to_numeric, errors="raise")
    if catalogue.isna().any().any() or not np.isfinite(catalogue.to_numpy()).all():
        raise ValueError("Candidate catalogue contains missing or infinite values")

    # Rounding removes harmless CSV floating-point spellings such as
    # 0.6000000000000001 before duplicate removal.
    catalogue = catalogue.round(10).drop_duplicates().reset_index(drop=True)
    if catalogue.empty:
        raise ValueError("Candidate catalogue is empty")

    support_checks = {
        "candidate_alpha": (ALPHA_MIN, ALPHA_MAX),
        "candidate_w_load_0": (W_LOAD_MIN, W_LOAD_MAX),
        "candidate_w_load_1": (W_LOAD_MIN, W_LOAD_MAX),
        "candidate_w_load_2": (W_LOAD_MIN, W_LOAD_MAX),
        "candidate_threshold_0": (THRESHOLD_MIN, THRESHOLD_MAX),
        "candidate_threshold_1": (THRESHOLD_MIN, THRESHOLD_MAX),
        "candidate_threshold_2": (THRESHOLD_MIN, THRESHOLD_MAX),
    }
    for column, (minimum, maximum) in support_checks.items():
        if not catalogue[column].between(minimum, maximum).all():
            raise ValueError(f"{column} contains values outside [{minimum}, {maximum}]")
    return catalogue


_validate_objective_configuration()
MANIFEST = _load_manifest()
MODELS = _load_models(MANIFEST)
FULL_CANDIDATE_CATALOGUE = _load_candidate_catalogue()
CANDIDATE_CATALOGUE, CATALOGUE_METADATA = select_catalogue(
    FULL_CANDIDATE_CATALOGUE, CANDIDATE_COLUMNS,
    float(os.getenv("SCOPE_CATALOGUE_FRACTION", "1.0")),
    int(os.getenv("SCOPE_CATALOGUE_SEED", "1")),
)
# Separate catalogue settings automatically; fresh root required per repeat.
CATALOGUE_TAG = (
    f"catalogue_{len(CANDIDATE_CATALOGUE)}_seed_"
    f"{CATALOGUE_METADATA['catalogue_subset_seed']}"
)
if DECISION_LOG_OVERRIDE:
    raise ValueError("Unset SCOPE_DECISION_LOG for catalogue experiments; use SCOPE_ABLATION_RESULTS_DIR")
ABLATION_RESULTS_DIR = ABLATION_RESULTS_DIR / CATALOGUE_TAG
ABLATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATE_CATALOGUE.to_csv(ABLATION_RESULTS_DIR / "selected_catalogue.csv", index=False)
with (ABLATION_RESULTS_DIR / "catalogue_metadata.json").open("w") as stream:
    json.dump(CATALOGUE_METADATA, stream, indent=2)
logger.info("Catalogue experiment: %s", CATALOGUE_METADATA)
logger.info(
    "Loaded %d/6 models and %d unique candidate configurations. "
    "lambda=%.2f, w_failure=%.2f, w_service=%.2f",
    len(MODELS),
    len(CANDIDATE_CATALOGUE),
    LAMBDA,
    W_FAILURE,
    W_SERVICE,
)


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _required(mapping: dict[str, Any], key: str, context: str) -> float:
    if key not in mapping:
        raise ValueError(f"Missing {context}.{key}")
    return _finite_number(mapping[key], f"{context}.{key}")


def _safe_rate(failed: float, sent: float) -> float:
    return failed / sent if sent > 0.0 else 0.0


def _parse_request_state(data: dict[str, Any]) -> tuple[dict[str, float], np.ndarray]:
    if "num_devices" not in data:
        raise ValueError(
            "Missing num_devices. Add it to VehicularEdgeOrchestrator.runOptimizer()."
        )
    num_devices = _finite_number(data["num_devices"], "num_devices")
    if num_devices <= 0:
        raise ValueError("num_devices must be positive")

    app_stats = data.get("app_stats")
    path_stats = data.get("path_stats")
    servers = data.get("servers")
    if not isinstance(app_stats, dict):
        raise ValueError("app_stats must be an object")
    if not isinstance(path_stats, dict):
        raise ValueError("path_stats must be an object")
    if not isinstance(servers, list) or not servers:
        raise ValueError("servers must be a non-empty list")

    server_loads = np.asarray(
        [_required(server, "load", f"servers[{index}]") for index, server in enumerate(servers)],
        dtype=float,
    )
    base: dict[str, float] = {
        "num_devices": num_devices,
        "state_avg_edge_cpu": float(np.mean(server_loads)),
        "state_max_edge_cpu": float(np.max(server_loads)),
    }

    task_totals = np.zeros(APP_COUNT, dtype=float)
    for app_id in range(APP_COUNT):
        key = str(app_id)
        stats = app_stats.get(key)
        if not isinstance(stats, dict):
            raise ValueError(f"Missing app_stats.{key}")

        total = _required(stats, "total", f"app_stats.{key}")
        lag = _required(stats, "lag", f"app_stats.{key}")
        completed = _required(stats, "completed", f"app_stats.{key}")
        failed = _required(stats, "failures", f"app_stats.{key}")
        failure_rate = _required(stats, "failure_rate", f"app_stats.{key}")
        mean_service_time = _required(
            stats, "mean_service_time", f"app_stats.{key}"
        )
        dnst = _required(
            stats, "deadline_norm_service_time", f"app_stats.{key}"
        )
        deadline_miss_rate = _required(
            stats, "deadline_miss_rate", f"app_stats.{key}"
        )

        base[f"state_total_{app_id}"] = total
        base[f"state_lag_total_{app_id}"] = lag
        base[f"state_completed_{app_id}"] = completed
        base[f"state_failed_{app_id}"] = failed
        base[f"state_failure_rate_{app_id}"] = failure_rate
        base[f"state_mean_service_time_{app_id}"] = mean_service_time
        base[f"state_deadline_norm_service_time_{app_id}"] = dnst
        base[f"state_deadline_miss_rate_{app_id}"] = deadline_miss_rate
        base[f"previous_w_load_{app_id}"] = _required(
            stats, "curr_w_load", f"app_stats.{key}"
        )
        base[f"previous_threshold_{app_id}"] = _required(
            stats, "curr_thr", f"app_stats.{key}"
        )
        task_totals[app_id] = max(total, 0.0)

    sent_edge = _required(path_stats, "sent_edge", "path_stats")
    sent_rsu = _required(path_stats, "sent_rsu", "path_stats")
    sent_gsm = _required(path_stats, "sent_gsm", "path_stats")
    fail_edge = _required(path_stats, "fail_edge", "path_stats")
    fail_rsu = _required(path_stats, "fail_rsu", "path_stats")
    fail_gsm = _required(path_stats, "fail_gsm", "path_stats")

    base.update(
        {
            "state_sent_edge": sent_edge,
            "state_sent_rsu": sent_rsu,
            "state_sent_gsm": sent_gsm,
            "state_fail_edge": fail_edge,
            "state_fail_rsu": fail_rsu,
            "state_fail_gsm": fail_gsm,
            # Model features use fractions, not percentages.
            "state_edge_failure_rate": _safe_rate(fail_edge, sent_edge),
            "state_rsu_failure_rate": _safe_rate(fail_rsu, sent_rsu),
            "state_gsm_failure_rate": _safe_rate(fail_gsm, sent_gsm),
            "previous_alpha": _required(path_stats, "curr_alpha", "path_stats"),
        }
    )

    if task_totals.sum() > 0.0:
        app_weights = task_totals / task_totals.sum()
    else:
        app_weights = np.full(APP_COUNT, 1.0 / APP_COUNT)
    return base, app_weights


def _current_candidate(base: dict[str, float]) -> dict[str, float]:
    candidate = {"candidate_alpha": base["previous_alpha"]}
    for app_id in range(APP_COUNT):
        candidate[f"candidate_w_load_{app_id}"] = base[f"previous_w_load_{app_id}"]
        candidate[f"candidate_threshold_{app_id}"] = base[
            f"previous_threshold_{app_id}"
        ]
    return candidate


def _candidate_batch(base: dict[str, float]) -> pd.DataFrame:
    current = pd.DataFrame([_current_candidate(base)])
    candidates = pd.concat([current, CANDIDATE_CATALOGUE], ignore_index=True)
    return candidates.round(10).drop_duplicates(CANDIDATE_COLUMNS).reset_index(drop=True)


def _feature_matrix(
    base: dict[str, float],
    candidates: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    matrix = pd.DataFrame(index=candidates.index)
    for feature in features:
        if feature in base:
            matrix[feature] = base[feature]
        elif feature in CANDIDATE_COLUMNS:
            matrix[feature] = candidates[feature].to_numpy()
        else:
            raise ValueError(f"Cannot construct model feature: {feature}")
    matrix = matrix[features].astype(np.float32)
    if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Constructed feature matrix contains invalid values")
    return matrix


def _predict_model(
    model_spec: dict[str, Any],
    base: dict[str, float],
    candidates: pd.DataFrame,
) -> np.ndarray:
    features = model_spec["features"]
    matrix = _feature_matrix(base, candidates, features)
    dmatrix = xgb.DMatrix(matrix, feature_names=features)
    best_iteration = int(model_spec["best_iteration"])
    return model_spec["booster"].predict(
        dmatrix,
        iteration_range=(0, best_iteration + 1),
        validate_features=True,
    )


def _reconfiguration_cost(
    base: dict[str, float], candidates: pd.DataFrame
) -> np.ndarray:
    # Each of the seven decision-variable changes is normalized by its feasible
    # range, then averaged. The result is dimensionless and approximately [0, 1].
    components = [
        np.abs(candidates["candidate_alpha"].to_numpy() - base["previous_alpha"])
        / (ALPHA_MAX - ALPHA_MIN)
    ]
    for app_id in range(APP_COUNT):
        components.append(
            np.abs(
                candidates[f"candidate_w_load_{app_id}"].to_numpy()
                - base[f"previous_w_load_{app_id}"]
            )
            / (W_LOAD_MAX - W_LOAD_MIN)
        )
        components.append(
            np.abs(
                candidates[f"candidate_threshold_{app_id}"].to_numpy()
                - base[f"previous_threshold_{app_id}"]
            )
            / (THRESHOLD_MAX - THRESHOLD_MIN)
        )
    return np.mean(np.vstack(components), axis=0)


def _score_candidates(
    base: dict[str, float],
    app_weights: np.ndarray,
    candidates: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, Any]:
    failure_predictions = np.zeros((len(candidates), APP_COUNT), dtype=float)
    dnst_predictions = np.zeros((len(candidates), APP_COUNT), dtype=float)

    for app_id in range(APP_COUNT):
        failure_predictions[:, app_id] = np.clip(
            _predict_model(MODELS[(app_id, "failure_rate")], base, candidates),
            0.0,
            1.0,
        )
        dnst_predictions[:, app_id] = np.clip(
            _predict_model(
                MODELS[(app_id, "deadline_norm_service_time")],
                base,
                candidates,
            ),
            0.0,
            None,
        )

    # Raw weighted predictions are logged in their interpretable units.
    weighted_failure = failure_predictions @ app_weights
    weighted_dnst = dnst_predictions @ app_weights

    # Objective terms are normalized using training-set support only.
    normalized_failure = np.clip(
        (failure_predictions - TRAIN_FAILURE_MIN)
        / (TRAIN_FAILURE_MAX - TRAIN_FAILURE_MIN),
        0.0,
        1.0,
    )
    normalized_dnst = np.clip(
        (dnst_predictions - TRAIN_DNST_MIN) / (TRAIN_DNST_MAX - TRAIN_DNST_MIN),
        0.0,
        1.0,
    )
    performance_cost = (
        W_FAILURE * (normalized_failure @ app_weights)
        + W_SERVICE * (normalized_dnst @ app_weights)
    )
    reconfiguration_cost = _reconfiguration_cost(base, candidates)
    # ===== EXPLICIT ABLATION MODES CHANGE START =====
    lambda_value = float(settings["lambda"])
    hard_enabled = bool(settings["hard_stability_enabled"])
    r_max = float(settings["r_max"])
    joint_cost = (
        (1.0 - lambda_value) * performance_cost
        + lambda_value * reconfiguration_cost
    )
    # ===== EXPLICIT ABLATION MODES CHANGE END =====

    # Primary ordering: joint cost. Ties prefer lower performance cost, then
    # lower churn. This keeps selection deterministic.
    # best_index = int(
    #     np.lexsort((reconfiguration_cost, performance_cost, joint_cost))[0]
    # )
    # ===== CHANGE START: APPLY HARD STABILITY CONSTRAINT =====

    # First identify the candidate selected by the soft objective alone.
    unconstrained_best_index = int(
        np.lexsort(
            (reconfiguration_cost, performance_cost, joint_cost)
        )[0]
    )

    if hard_enabled:
        feasible_mask = reconfiguration_cost <= r_max + 1e-12
    else:
        feasible_mask = np.ones(len(candidates), dtype=bool)

    # The current configuration has R=0, so at least one candidate
    # should always remain feasible.
    if not np.any(feasible_mask):
        raise RuntimeError(
            "No candidate satisfies the hard stability constraint"
        )

    # Infeasible candidates cannot be selected.
    constrained_joint_cost = np.where(
        feasible_mask,
        joint_cost,
        np.inf,
    )

    # Select the best candidate among all allowed candidates.
    best_index = int(
        np.lexsort(
            (
                reconfiguration_cost,
                performance_cost,
                constrained_joint_cost,
            )
        )[0]
    )

    constraint_active = bool(
        hard_enabled
        and not feasible_mask[unconstrained_best_index]
    )

    # ===== CHANGE END: APPLY HARD STABILITY CONSTRAINT =====
    return {
        "best_index": best_index,
        "failure_predictions": failure_predictions,
        "dnst_predictions": dnst_predictions,
        "weighted_failure": weighted_failure,
        "weighted_dnst": weighted_dnst,
        "performance_cost": performance_cost,
        "reconfiguration_cost": reconfiguration_cost,
        "joint_cost": joint_cost,
        # ===== CHANGE START: RETURN STABILITY INFORMATION =====

        "feasible_candidate_count": int(np.sum(feasible_mask)),
        "constraint_active": constraint_active,
        "unconstrained_best_index": unconstrained_best_index,

        # ===== CHANGE END: RETURN STABILITY INFORMATION =====
    }


def _response_string(candidate: pd.Series) -> str:
    sections = [f"ALPHA:{candidate['candidate_alpha']:.2f}"]
    for app_id in range(APP_COUNT):
        w_load = float(candidate[f"candidate_w_load_{app_id}"])
        threshold = float(candidate[f"candidate_threshold_{app_id}"])
        sections.append(
            f"{app_id}:{w_load:.2f}:{1.0 - w_load:.2f}:{threshold:.1f}"
        )
    return "|".join(sections) + "|"


def _write_decision_log(
    data: dict[str, Any],
    base: dict[str, float],
    app_weights: np.ndarray,
    candidates: pd.DataFrame,
    scores: dict[str, Any],
    settings: dict[str, Any],
    optimizer_time_ms: float,
) -> None:
    index = scores["best_index"]
    selected = candidates.iloc[index]
    row: dict[str, Any] = {
        # ===== ABLATION RUN IDENTITY CHANGE START =====
        "variant": settings["variant"],
        "policy": settings["policy"],
        "run_id": data.get("run_id", ""),
        "iteration": data.get("iteration", ""),
        "simulation_seed": data.get("simulation_seed", ""),
        # ===== ABLATION RUN IDENTITY CHANGE END =====
        "timestamp": data.get("timestamp", ""),
        "num_devices": data.get("num_devices", ""),
        "lambda": settings["lambda"],
        "w_failure": W_FAILURE,
        "w_service": W_SERVICE,
        "candidate_count": len(candidates),
        **CATALOGUE_METADATA,
        "optimizer_time_ms": optimizer_time_ms,
        "previous_alpha": base["previous_alpha"],
        "selected_alpha": selected["candidate_alpha"],
        "weighted_predicted_failure_rate": scores["weighted_failure"][index],
        "weighted_predicted_dnst": scores["weighted_dnst"][index],
        "performance_cost": scores["performance_cost"][index],
        "reconfiguration_cost": scores["reconfiguration_cost"][index],
        "joint_cost": scores["joint_cost"][index],
        # ===== CHANGE START: LOG HARD STABILITY INFORMATION =====

        "hard_stability_enabled": settings["hard_stability_enabled"],
        "r_max": settings["r_max"],
        "feasible_candidate_count": scores[
            "feasible_candidate_count"
        ],
        "constraint_active": scores["constraint_active"],
        "reconfigured": bool(scores["reconfiguration_cost"][index] > 1e-12),
        "stability_violation": bool(
            scores["reconfiguration_cost"][index]
            > float(settings["r_max"]) + 1e-12
        ),

        # ===== CHANGE END: LOG HARD STABILITY INFORMATION =====
        # ===== CHANGE START: LOG UNCONSTRAINED CANDIDATE =====

        "unconstrained_reconfiguration_cost": scores[
            "reconfiguration_cost"
        ][scores["unconstrained_best_index"]],

        "unconstrained_performance_cost": scores[
            "performance_cost"
        ][scores["unconstrained_best_index"]],

        # ===== CHANGE END: LOG UNCONSTRAINED CANDIDATE =====
    }

    for app_id in range(APP_COUNT):
        row[f"app_weight_{app_id}"] = app_weights[app_id]
        row[f"previous_w_load_{app_id}"] = base[
            f"previous_w_load_{app_id}"
        ]
        row[f"previous_threshold_{app_id}"] = base[
            f"previous_threshold_{app_id}"
        ]
        row[f"selected_w_load_{app_id}"] = selected[
            f"candidate_w_load_{app_id}"
        ]
        row[f"selected_threshold_{app_id}"] = selected[
            f"candidate_threshold_{app_id}"
        ]
        row[f"predicted_failure_rate_{app_id}"] = scores[
            "failure_predictions"
        ][index, app_id]
        row[f"predicted_dnst_{app_id}"] = scores["dnst_predictions"][
            index, app_id
        ]

    # ===== ABLATION-SAFE DECISION LOGGING CHANGE START =====
    decision_log_path = _decision_log_path(data, str(settings["variant"]))
    decision_log_path.parent.mkdir(parents=True, exist_ok=True)
    with decision_log_lock:
        write_header = (
            not decision_log_path.exists()
            or decision_log_path.stat().st_size == 0
        )
        with decision_log_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    # ===== ABLATION-SAFE DECISION LOGGING CHANGE END =====


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ready",
            "models_loaded": len(MODELS),
            "candidate_count": len(CANDIDATE_CATALOGUE),
            **CATALOGUE_METADATA,
            "lambda": LAMBDA,
            "w_failure": W_FAILURE,
            "w_service": W_SERVICE,
            # ===== EXPLICIT ABLATION MODES CHANGE START =====
            "supported_ablation_policies": [
                "SCOPE",
                "SOFTCOST_OPT",
                "PERFORMANCE_ONLY",
                "RT_OPT_ML",
            ],
            "automatic_decision_log_root": str(ABLATION_RESULTS_DIR),
            # ===== EXPLICIT ABLATION MODES CHANGE END =====
            # ===== CHANGE START: REPORT HARD STABILITY STATUS =====

            "hard_stability_enabled": HARD_STABILITY_ENABLED,
            "r_max": R_MAX,

            # ===== CHANGE END: REPORT HARD STABILITY STATUS =====
        }
    )


@app.route("/optimize", methods=["POST"])
def optimize():
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object")

        # ===== EXPLICIT ABLATION MODES CHANGE START =====
        settings = _resolve_objective_settings(data)
        optimizer_start = time.perf_counter()
        # ===== EXPLICIT ABLATION MODES CHANGE END =====
        base, app_weights = _parse_request_state(data)
        candidates = _candidate_batch(base)
        scores = _score_candidates(base, app_weights, candidates, settings)
        selected = candidates.iloc[scores["best_index"]]
        optimizer_time_ms = (time.perf_counter() - optimizer_start) * 1000.0
        _write_decision_log(
            data,
            base,
            app_weights,
            candidates,
            scores,
            settings,
            optimizer_time_ms,
        )

        logger.info(
            "variant=%s time=%s devices=%s candidates=%d alpha %.2f->%.2f "
            "pred_failure=%.4f pred_dnst=%.4f performance=%.4f "
            "reconfiguration=%.4f joint=%.4f optimizer_ms=%.3f",
            settings["variant"],
            data.get("timestamp", "?"),
            data.get("num_devices", "?"),
            len(candidates),
            base["previous_alpha"],
            selected["candidate_alpha"],
            scores["weighted_failure"][scores["best_index"]],
            scores["weighted_dnst"][scores["best_index"]],
            scores["performance_cost"][scores["best_index"]],
            scores["reconfiguration_cost"][scores["best_index"]],
            scores["joint_cost"][scores["best_index"]],
            optimizer_time_ms,
        )
        return _response_string(selected)

    except Exception as exc:
        # Do not silently substitute CPU load for an invalid prediction. Java
        # will retain the previous configuration when this request fails.
        logger.exception("Optimization failed: %s", exc)
        return "ERROR", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)


# ===== SCOPE JOINT OBJECTIVE CHANGE END =====
