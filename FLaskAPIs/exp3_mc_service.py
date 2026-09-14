"""EXP3-MC-adapted external baseline for the SCOPE EdgeCloudSim study.

The implementation follows Donassolo et al., IEEE TPDS 2022:

* an EXP3 exponential-weight learner uses bandit feedback;
* the learner observes only the reward of the active configuration;
* migration/reconfiguration is reactive and is attempted only when the
  observed performance cost exceeds a configured QoS threshold.

Adaptation to SCOPE:

* each arm is one complete SCOPE configuration
  (alpha, three load weights, and three thresholds);
* five arms are reproducibly drawn from the SCOPE candidate catalogue,
  matching the five-candidate decision size evaluated in the paper;
* observed reward is one minus SCOPE's normalized performance cost, without
  a churn penalty and without a hard R_max constraint;
* the paper's eta=0.1 is retained by default.

Run with:

    python exp3_mc_service.py

Required data:

    scope_candidate_dataset/candidate_dataset_train.csv

or set EXP3_MC_CANDIDATE_DATASET to its absolute path.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


BASE_DIR = Path(__file__).resolve().parent

# ===== EXP3-MC ADAPTED CHANGE START: CONFIGURATION =====
CANDIDATE_DATASET_PATH = Path(
    os.getenv(
        "EXP3_MC_CANDIDATE_DATASET",
        str(BASE_DIR / "scope_candidate_dataset" / "candidate_dataset_train.csv"),
    )
)
# ===== EXP3-MC ADAPTED CHANGE START: AUTOMATIC RESULT FILENAMES =====
# EXP3_MC_DECISION_LOG remains an optional exact-path override for tests or
# special runs. Normally the service derives a unique file from the QoS value,
# eta, run ID, simulation seed, and device count sent by EdgeCloudSim.
DECISION_LOG_OVERRIDE = os.getenv("EXP3_MC_DECISION_LOG")
RESULTS_DIR = Path(
    os.getenv("EXP3_MC_RESULTS_DIR", str(BASE_DIR / "exp3_mc_results"))
)
# ===== EXP3-MC ADAPTED CHANGE END: AUTOMATIC RESULT FILENAMES =====

QOS_THRESHOLD = float(os.getenv("EXP3_MC_QOS_THRESHOLD", "0.20"))
W_FAILURE = float(os.getenv("EXP3_MC_W_FAILURE", "0.50"))
W_SERVICE = float(os.getenv("EXP3_MC_W_SERVICE", "0.50"))
ETA = float(os.getenv("EXP3_MC_ETA", "0.10"))
ARM_COUNT = int(os.getenv("EXP3_MC_ARM_COUNT", "5"))
ARM_SELECTION_SEED = int(os.getenv("EXP3_MC_ARM_SELECTION_SEED", "20260828"))
SEED_OFFSET = int(os.getenv("EXP3_MC_SEED_OFFSET", "300007"))
PORT = int(os.getenv("EXP3_MC_PORT", "5003"))

APP_COUNT = 3
CANDIDATE_COLUMNS = [
    "candidate_alpha",
    "candidate_w_load_0",
    "candidate_w_load_1",
    "candidate_w_load_2",
    "candidate_threshold_0",
    "candidate_threshold_1",
    "candidate_threshold_2",
]

# Same training-only normalization support used by SCOPE.
TRAIN_FAILURE_MIN = 0.0
TRAIN_FAILURE_MAX = 0.7353648757016841
TRAIN_DNST_MIN = 0.5327461065548065
TRAIN_DNST_MAX = 1.6538610651089858

ALPHA_MIN, ALPHA_MAX = 0.05, 0.95
W_LOAD_MIN, W_LOAD_MAX = 0.10, 0.90
THRESHOLD_MIN, THRESHOLD_MAX = 20.0, 95.0
# ===== EXP3-MC ADAPTED CHANGE END: CONFIGURATION =====


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [EXP3-MC-ADAPTED] %(message)s",
)
logger = logging.getLogger(__name__)
app = Flask(__name__)
state_lock = threading.Lock()
decision_log_lock = threading.Lock()


def _validate_configuration() -> None:
    if not 0.0 <= QOS_THRESHOLD <= 1.0:
        raise ValueError("EXP3_MC_QOS_THRESHOLD must be between 0 and 1")
    if W_FAILURE < 0.0 or W_SERVICE < 0.0:
        raise ValueError("EXP3-MC performance weights cannot be negative")
    if not math.isclose(W_FAILURE + W_SERVICE, 1.0, abs_tol=1e-9):
        raise ValueError(
            "EXP3_MC_W_FAILURE + EXP3_MC_W_SERVICE must equal 1"
        )
    if ETA <= 0.0:
        raise ValueError("EXP3_MC_ETA must be positive")
    if ARM_COUNT < 2:
        raise ValueError("EXP3_MC_ARM_COUNT must be at least 2")


def _load_candidate_catalogue() -> pd.DataFrame:
    if not CANDIDATE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing candidate dataset: {CANDIDATE_DATASET_PATH}. "
            "Set EXP3_MC_CANDIDATE_DATASET to the same training candidate "
            "CSV used by SCOPE."
        )

    candidates = pd.read_csv(
        CANDIDATE_DATASET_PATH,
        usecols=CANDIDATE_COLUMNS,
    )
    candidates = (
        candidates.astype(float)
        .round(10)
        .drop_duplicates(CANDIDATE_COLUMNS)
        .reset_index(drop=True)
    )
    if candidates.empty:
        raise ValueError("The EXP3-MC candidate catalogue is empty")
    if candidates.isna().any().any():
        raise ValueError("The EXP3-MC candidate catalogue contains NaN values")

    bounds = {
        "candidate_alpha": (ALPHA_MIN, ALPHA_MAX),
        "candidate_w_load_0": (W_LOAD_MIN, W_LOAD_MAX),
        "candidate_w_load_1": (W_LOAD_MIN, W_LOAD_MAX),
        "candidate_w_load_2": (W_LOAD_MIN, W_LOAD_MAX),
        "candidate_threshold_0": (THRESHOLD_MIN, THRESHOLD_MAX),
        "candidate_threshold_1": (THRESHOLD_MIN, THRESHOLD_MAX),
        "candidate_threshold_2": (THRESHOLD_MIN, THRESHOLD_MAX),
    }
    for column, (lower, upper) in bounds.items():
        if not candidates[column].between(lower, upper).all():
            raise ValueError(f"Candidate column {column} is outside its support")
    return candidates


_validate_configuration()
CANDIDATE_CATALOGUE = _load_candidate_catalogue()


@dataclass
class Exp3McState:
    candidates: pd.DataFrame
    logits: np.ndarray
    current_index: int
    rng: np.random.Generator
    eta: float
    last_timestamp: float = -1.0
    windows: int = 0
    reconfigurations: int = 0
    total_churn: float = 0.0


STATES: dict[str, Exp3McState] = {}


def _required(mapping: dict[str, Any], key: str, context: str) -> float:
    if key not in mapping:
        raise ValueError(f"Missing {context}.{key}")
    value = float(mapping[key])
    if not math.isfinite(value):
        raise ValueError(f"Invalid {context}.{key}: {value}")
    return value


def _current_candidate(data: dict[str, Any]) -> dict[str, float]:
    path_stats = data.get("path_stats")
    app_stats = data.get("app_stats")
    if not isinstance(path_stats, dict):
        raise ValueError("path_stats must be an object")
    if not isinstance(app_stats, dict):
        raise ValueError("app_stats must be an object")

    candidate = {
        "candidate_alpha": _required(path_stats, "curr_alpha", "path_stats")
    }
    for app_id in range(APP_COUNT):
        stats = app_stats.get(str(app_id))
        if not isinstance(stats, dict):
            raise ValueError(f"Missing app_stats.{app_id}")
        candidate[f"candidate_w_load_{app_id}"] = _required(
            stats, "curr_w_load", f"app_stats.{app_id}"
        )
        candidate[f"candidate_threshold_{app_id}"] = _required(
            stats, "curr_thr", f"app_stats.{app_id}"
        )
    return candidate


def _candidate_batch(current: dict[str, float]) -> pd.DataFrame:
    """Build one fixed, reproducible arm set from the common catalogue."""
    current_row = pd.DataFrame([current], columns=CANDIDATE_COLUMNS)
    all_candidates = (
        pd.concat([current_row, CANDIDATE_CATALOGUE], ignore_index=True)
        .astype(float)
        .round(10)
        .drop_duplicates(CANDIDATE_COLUMNS)
        .reset_index(drop=True)
    )

    # ===== EXP3-MC ADAPTED CHANGE START: FIVE REPRODUCIBLE ARMS =====
    # Donassolo et al. evaluate five randomly selected candidate hosts. Here,
    # arm zero is the simulator's initial configuration and the remaining arms
    # are sampled once from the common feasible SCOPE catalogue with a fixed,
    # policy-independent seed. Keeping this subset fixed across iterations
    # prevents the arm set itself from becoming an experimental confound.
    if len(all_candidates) <= ARM_COUNT:
        return all_candidates

    selection_rng = np.random.default_rng(ARM_SELECTION_SEED)
    selected_tail = np.sort(
        selection_rng.choice(
            np.arange(1, len(all_candidates)),
            size=ARM_COUNT - 1,
            replace=False,
        )
    )
    selected_indices = np.concatenate(([0], selected_tail))
    return all_candidates.iloc[selected_indices].reset_index(drop=True)
    # ===== EXP3-MC ADAPTED CHANGE END: FIVE REPRODUCIBLE ARMS =====


def _find_candidate_index(
    candidates: pd.DataFrame, candidate: dict[str, float]
) -> int:
    target = np.asarray([candidate[column] for column in CANDIDATE_COLUMNS])
    matrix = candidates[CANDIDATE_COLUMNS].to_numpy(dtype=float)
    matches = np.all(np.isclose(matrix, target, atol=1e-9, rtol=0.0), axis=1)
    indices = np.flatnonzero(matches)
    if len(indices) != 1:
        raise ValueError("Current configuration is missing or duplicated")
    return int(indices[0])


def _parse_observed_performance(
    data: dict[str, Any],
) -> tuple[float, float, float, np.ndarray]:
    app_stats = data.get("app_stats")
    if not isinstance(app_stats, dict):
        raise ValueError("app_stats must be an object")

    totals = np.zeros(APP_COUNT, dtype=float)
    failure_rates = np.zeros(APP_COUNT, dtype=float)
    dnst = np.zeros(APP_COUNT, dtype=float)

    for app_id in range(APP_COUNT):
        stats = app_stats.get(str(app_id))
        if not isinstance(stats, dict):
            raise ValueError(f"Missing app_stats.{app_id}")
        totals[app_id] = max(
            0.0, _required(stats, "total", f"app_stats.{app_id}")
        )
        failure_rates[app_id] = np.clip(
            _required(stats, "failure_rate", f"app_stats.{app_id}"),
            0.0,
            1.0,
        )
        dnst[app_id] = max(
            0.0,
            _required(
                stats,
                "deadline_norm_service_time",
                f"app_stats.{app_id}",
            ),
        )

    if totals.sum() > 0.0:
        app_weights = totals / totals.sum()
    else:
        app_weights = np.full(APP_COUNT, 1.0 / APP_COUNT)

    normalized_failure = np.clip(
        (failure_rates - TRAIN_FAILURE_MIN)
        / (TRAIN_FAILURE_MAX - TRAIN_FAILURE_MIN),
        0.0,
        1.0,
    )
    normalized_dnst = np.clip(
        (dnst - TRAIN_DNST_MIN) / (TRAIN_DNST_MAX - TRAIN_DNST_MIN),
        0.0,
        1.0,
    )
    weighted_failure = float(failure_rates @ app_weights)
    weighted_dnst = float(dnst @ app_weights)
    performance_cost = float(
        W_FAILURE * (normalized_failure @ app_weights)
        + W_SERVICE * (normalized_dnst @ app_weights)
    )
    return performance_cost, weighted_failure, weighted_dnst, app_weights


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / weights.sum()


# ===== EXP3-MC ADAPTED CHANGE START: COMMON NORMALIZED CHURN METRIC =====
def _reconfiguration_cost(
    first: dict[str, float], second: pd.Series
) -> float:
    """Use the exact seven-component normalization used by SCOPE."""
    components = [
        abs(float(second["candidate_alpha"]) - first["candidate_alpha"])
        / (ALPHA_MAX - ALPHA_MIN)
    ]
    for app_id in range(APP_COUNT):
        components.append(
            abs(
                float(second[f"candidate_w_load_{app_id}"])
                - first[f"candidate_w_load_{app_id}"]
            )
            / (W_LOAD_MAX - W_LOAD_MIN)
        )
        components.append(
            abs(
                float(second[f"candidate_threshold_{app_id}"])
                - first[f"candidate_threshold_{app_id}"]
            )
            / (THRESHOLD_MAX - THRESHOLD_MIN)
        )
    return float(np.mean(components))
# ===== EXP3-MC ADAPTED CHANGE END: COMMON NORMALIZED CHURN METRIC =====


def _run_key(data: dict[str, Any]) -> str:
    run_id = str(data.get("run_id", "run_unspecified"))
    simulation_seed = int(float(data.get("simulation_seed", 20260828)))
    num_devices = int(float(data.get("num_devices", 0)))
    return f"{run_id}|seed={simulation_seed}|devices={num_devices}"


def _new_state(data: dict[str, Any], current: dict[str, float]) -> Exp3McState:
    candidates = _candidate_batch(current)
    current_index = _find_candidate_index(candidates, current)
    arm_count = len(candidates)
    simulation_seed = int(float(data.get("simulation_seed", 20260828)))
    rng = np.random.default_rng(simulation_seed + SEED_OFFSET)
    return Exp3McState(
        candidates=candidates,
        logits=np.zeros(arm_count, dtype=float),
        current_index=current_index,
        rng=rng,
        eta=ETA,
    )


def _response_string(candidate: pd.Series) -> str:
    sections = [f"ALPHA:{candidate['candidate_alpha']:.2f}"]
    for app_id in range(APP_COUNT):
        w_load = float(candidate[f"candidate_w_load_{app_id}"])
        threshold = float(candidate[f"candidate_threshold_{app_id}"])
        sections.append(
            f"{app_id}:{w_load:.2f}:{1.0 - w_load:.2f}:{threshold:.1f}"
        )
    return "|".join(sections)


def _decision_log_path(row: dict[str, Any]) -> Path:
    if DECISION_LOG_OVERRIDE:
        return Path(DECISION_LOG_OVERRIDE)
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["run_id"]))
    qos_label = f"{QOS_THRESHOLD:.2f}".replace(".", "")
    eta_label = f"{ETA:.2f}".replace(".", "")
    return RESULTS_DIR / (
        f"exp3_mc_qos{qos_label}_eta{eta_label}_{safe_run_id}_"
        f"seed{row['simulation_seed']}_n{row['num_devices']}.csv"
    )


def _write_decision_log(row: dict[str, Any]) -> None:
    decision_log_path = _decision_log_path(row)
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


def _decision(data: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    timestamp = float(data.get("timestamp", 0.0))
    current = _current_candidate(data)
    performance_cost, weighted_failure, weighted_dnst, app_weights = (
        _parse_observed_performance(data)
    )
    reward = float(np.clip(1.0 - performance_cost, 0.0, 1.0))
    key = _run_key(data)

    with state_lock:
        state = STATES.get(key)
        # A non-increasing timestamp identifies a repeated/new simulation using
        # the same run key, so stale bandit state must not leak across runs.
        if state is None or timestamp <= state.last_timestamp:
            state = _new_state(data, current)
            STATES[key] = state
            reset = True
        else:
            reset = False
            observed_index = _find_candidate_index(state.candidates, current)
            if observed_index != state.current_index:
                logger.warning(
                    "Recovering current arm for %s: expected %d, observed %d",
                    key,
                    state.current_index,
                    observed_index,
                )
                state.current_index = observed_index

        probabilities_before = _softmax(state.logits)
        active_probability = max(
            float(probabilities_before[state.current_index]), 1e-12
        )

        # Paper equations (3)-(4): importance-weighted bandit feedback updates
        # only the active arm, since no counterfactual rewards are available.
        estimated_reward = reward / active_probability
        state.logits[state.current_index] += state.eta * estimated_reward
        state.logits -= np.max(state.logits)
        probabilities_after = _softmax(state.logits)

        triggered = performance_cost > QOS_THRESHOLD
        previous_index = state.current_index
        if triggered:
            selected_index = int(
                state.rng.choice(len(state.candidates), p=probabilities_after)
            )
        else:
            selected_index = previous_index

        switched = selected_index != previous_index
        selected = state.candidates.iloc[selected_index]
        step_churn = _reconfiguration_cost(current, selected)
        if switched:
            state.reconfigurations += 1
        state.total_churn += step_churn
        state.current_index = selected_index
        state.last_timestamp = timestamp
        state.windows += 1

        row: dict[str, Any] = {
            "run_key": key,
            "run_id": str(data.get("run_id", "run_unspecified")),
            "simulation_seed": int(
                float(data.get("simulation_seed", 20260828))
            ),
            "num_devices": int(float(data.get("num_devices", 0))),
            "timestamp": timestamp,
            "reset": reset,
            "window": state.windows,
            "arm_count": len(state.candidates),
            "eta": state.eta,
            "qos_threshold": QOS_THRESHOLD,
            "performance_cost": performance_cost,
            "reward": reward,
            "weighted_failure_rate": weighted_failure,
            "weighted_deadline_norm_service_time": weighted_dnst,
            "triggered": triggered,
            "previous_arm": previous_index,
            "selected_arm": selected_index,
            "selected_probability": probabilities_after[selected_index],
            "switched": switched,
            "reconfiguration_count": state.reconfigurations,
            "step_churn": step_churn,
            "total_churn": state.total_churn,
        }
        for app_id in range(APP_COUNT):
            row[f"app_weight_{app_id}"] = app_weights[app_id]
        for column in CANDIDATE_COLUMNS:
            row[f"selected_{column.removeprefix('candidate_')}"] = selected[
                column
            ]
        return selected, row


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "baseline": "EXP3-MC-adapted",
            "candidate_count": len(CANDIDATE_CATALOGUE),
            "arm_count": ARM_COUNT,
            "arm_selection_seed": ARM_SELECTION_SEED,
            "qos_threshold": QOS_THRESHOLD,
            "eta": ETA,
            "state_count": len(STATES),
        }
    )


@app.post("/reset")
def reset():
    with state_lock:
        STATES.clear()
    return jsonify({"status": "reset"})


@app.post("/optimize")
def optimize():
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            raise ValueError("Request JSON must be an object")
        selected, row = _decision(data)
        _write_decision_log(row)
        logger.info(
            "run=%s time=%.1f cost=%.4f reward=%.4f trigger=%s "
            "arm=%d->%d switched=%s reconfigurations=%d",
            row["run_key"],
            row["timestamp"],
            row["performance_cost"],
            row["reward"],
            row["triggered"],
            row["previous_arm"],
            row["selected_arm"],
            row["switched"],
            row["reconfiguration_count"],
        )
        return jsonify(_response_string(selected))
    except Exception as exc:  # Java receives a non-2xx response on failure.
        logger.exception("EXP3-MC optimization failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


logger.info(
    "Loaded %d unique catalogue configurations; using at most %d arms "
    "selected with seed %d. QoS threshold=%.3f, "
    "weights=(failure=%.2f, service=%.2f), eta=%.3f",
    len(CANDIDATE_CATALOGUE),
    ARM_COUNT,
    ARM_SELECTION_SEED,
    QOS_THRESHOLD,
    W_FAILURE,
    W_SERVICE,
    ETA,
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
