"""RT_OPT non-ML baseline service.

This file deliberately remains a measured-state, non-predictive baseline on
port 5001. It is not the SCOPE joint optimizer. The changes below only make its
feasible ranges, failure-rate calculation, and boundary handling consistent
and safe for comparison with SCOPE.
"""

import csv
import logging
import os
import re
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request
from ortools.sat.python import cp_model


app = Flask(__name__)
PORT = 5001

# ===== RT-OPT ABLATION LOGGING CHANGE START =====
APP_COUNT = 3
BASE_DIR = Path(__file__).resolve().parent
DECISION_LOG_OVERRIDE = os.getenv("RT_OPT_DECISION_LOG", "").strip()
ABLATION_RESULTS_DIR = Path(
    os.getenv(
        "SCOPE_ABLATION_RESULTS_DIR",
        str(BASE_DIR / "scope_ablation_results"),
    )
)
R_MAX_REFERENCE = float(os.getenv("SCOPE_R_MAX", "0.15"))
decision_log_lock = threading.Lock()
# ===== RT-OPT ABLATION LOGGING CHANGE END =====

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [RT-OPT] %(message)s")

# ===== RT_OPT BASELINE SAFETY CHANGE START: shared feasible configuration range =====
ALPHA_MIN = 0.05
ALPHA_MAX = 0.95
W_LOAD_MIN_PERCENT = 10
W_LOAD_MAX_PERCENT = 90
THRESHOLD_MIN = 20
THRESHOLD_MAX = 95
# ===== RT_OPT BASELINE SAFETY CHANGE END =====


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


# ===== RT-OPT ABLATION LOGGING CHANGE START =====
def _safe_file_token(value, fallback):
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    token = token.strip("._-")
    return token or fallback


def _decision_log_path(data):
    if DECISION_LOG_OVERRIDE:
        return Path(DECISION_LOG_OVERRIDE)

    run_id = _safe_file_token(data.get("run_id", "run_unspecified"), "run_unspecified")
    iteration = _safe_file_token(data.get("iteration", "unknown"), "unknown")
    seed = _safe_file_token(data.get("simulation_seed", "unknown"), "unknown")
    devices = _safe_file_token(data.get("num_devices", "unknown"), "unknown")
    filename = f"rt_opt_iteration{iteration}_seed{seed}_n{devices}_{run_id}.csv"
    return ABLATION_RESULTS_DIR / "decision_logs" / filename


def _normalized_reconfiguration_cost(
    old_alpha,
    new_alpha,
    old_w_load,
    new_w_load,
    old_threshold,
    new_threshold,
):
    """Use exactly the same seven-component normalization as SCOPE."""
    components = [abs(new_alpha - old_alpha) / (ALPHA_MAX - ALPHA_MIN)]
    for app_id in range(APP_COUNT):
        components.append(
            abs(new_w_load[app_id] - old_w_load[app_id])
            / ((W_LOAD_MAX_PERCENT - W_LOAD_MIN_PERCENT) / 100.0)
        )
        components.append(
            abs(new_threshold[app_id] - old_threshold[app_id])
            / (THRESHOLD_MAX - THRESHOLD_MIN)
        )
    return sum(components) / len(components)


def _write_decision_log(
    data,
    old_alpha,
    new_alpha,
    old_w_load,
    new_w_load,
    old_threshold,
    new_threshold,
    reconfiguration_cost,
    optimizer_time_ms,
    solver_status,
):
    row = {
        "variant": "RT_OPT",
        "policy": str(data.get("policy", "RT_OPT")),
        "run_id": data.get("run_id", ""),
        "iteration": data.get("iteration", ""),
        "simulation_seed": data.get("simulation_seed", ""),
        "timestamp": data.get("timestamp", ""),
        "num_devices": data.get("num_devices", ""),
        "lambda": "",
        "w_failure": "",
        "w_service": "",
        "candidate_count": "",
        "optimizer_time_ms": optimizer_time_ms,
        "previous_alpha": old_alpha,
        "selected_alpha": new_alpha,
        "weighted_predicted_failure_rate": "",
        "weighted_predicted_dnst": "",
        "performance_cost": "",
        "reconfiguration_cost": reconfiguration_cost,
        "joint_cost": "",
        "hard_stability_enabled": False,
        "r_max": R_MAX_REFERENCE,
        "feasible_candidate_count": "",
        "constraint_active": False,
        "reconfigured": bool(reconfiguration_cost > 1e-12),
        "stability_violation": bool(
            reconfiguration_cost > R_MAX_REFERENCE + 1e-12
        ),
        "unconstrained_reconfiguration_cost": reconfiguration_cost,
        "unconstrained_performance_cost": "",
        "solver_status": solver_status,
    }
    for app_id in range(APP_COUNT):
        row[f"app_weight_{app_id}"] = ""
        row[f"previous_w_load_{app_id}"] = old_w_load[app_id]
        row[f"previous_threshold_{app_id}"] = old_threshold[app_id]
        row[f"selected_w_load_{app_id}"] = new_w_load[app_id]
        row[f"selected_threshold_{app_id}"] = new_threshold[app_id]
        row[f"predicted_failure_rate_{app_id}"] = ""
        row[f"predicted_dnst_{app_id}"] = ""

    decision_log_path = _decision_log_path(data)
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


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ready",
            "variant": "RT_OPT",
            "automatic_decision_log_root": str(ABLATION_RESULTS_DIR),
            "r_max_reference": R_MAX_REFERENCE,
        }
    )
# ===== RT-OPT ABLATION LOGGING CHANGE END =====


@app.route("/optimize", methods=["POST"])
def optimize():
    try:
        # ===== RT-OPT ABLATION LOGGING CHANGE START =====
        optimizer_start = time.perf_counter()
        # ===== RT-OPT ABLATION LOGGING CHANGE END =====
        data = request.get_json(force=True)
        timestamp = data.get("timestamp")
        app_stats = data.get("app_stats", {})
        path_stats = data.get("path_stats", {})

        # 1. Parse currently measured path statistics.
        sent_rsu = float(path_stats.get("sent_rsu", 0))
        sent_gsm = float(path_stats.get("sent_gsm", 0))
        fail_rsu = float(path_stats.get("fail_rsu", 0))
        fail_gsm = float(path_stats.get("fail_gsm", 0))
        curr_alpha = clamp(
            float(path_stats.get("curr_alpha", 0.6)),
            ALPHA_MIN,
            ALPHA_MAX,
        )

        rate_rsu = (fail_rsu / sent_rsu * 100.0) if sent_rsu > 0 else 0.0
        rate_gsm = (fail_gsm / sent_gsm * 100.0) if sent_gsm > 0 else 0.0

        logging.info("--- TIME %ss ---", timestamp)
        logging.info(
            "PATH STATS: RSU_Fail=%.1f%% (%s/%s) | GSM_Fail=%.1f%% (%s/%s)",
            rate_rsu,
            fail_rsu,
            sent_rsu,
            rate_gsm,
            fail_gsm,
            sent_gsm,
        )

        # 2. Non-predictive alpha heuristic. This remains intentionally based
        # only on currently observed path failures because RT_OPT is the no-ML
        # internal baseline.
        new_alpha = curr_alpha
        if rate_gsm > rate_rsu + 5.0:
            logging.info("GSM is failing; shifting traffic toward RSU.")
            new_alpha = curr_alpha + 0.15
        elif rate_rsu > rate_gsm + 5.0:
            logging.info("RSU is failing; shifting traffic toward GSM.")
            new_alpha = curr_alpha - 0.15
        elif rate_gsm > 10.0 and rate_rsu > 10.0 and rate_rsu < rate_gsm:
            new_alpha = curr_alpha + 0.05

        # ===== RT_OPT BASELINE SAFETY CHANGE START: clamp and snap alpha =====
        new_alpha = clamp(new_alpha, ALPHA_MIN, ALPHA_MAX)
        new_alpha = round(new_alpha / 0.05) * 0.05
        new_alpha = clamp(new_alpha, ALPHA_MIN, ALPHA_MAX)
        # ===== RT_OPT BASELINE SAFETY CHANGE END =====

        # 3. RT_OPT edge-parameter model. Measured failure conditions are hard
        # constraints; the objective remains reconfiguration minimization.
        model = cp_model.CpModel()
        variables = {}
        churn_terms = []
        # ===== RT-OPT ABLATION LOGGING CHANGE START =====
        old_w_load_values = {}
        old_threshold_values = {}
        # ===== RT-OPT ABLATION LOGGING CHANGE END =====

        for type_id, stats in app_stats.items():
            app_id = int(type_id)
            failures = float(stats.get("failures", 0))

            # ===== RT_OPT BASELINE SAFETY CHANGE START: finalized-task denominator =====
            # Failure rate must use completed + failed tasks. Dividing by all
            # generated tasks incorrectly treats unfinished tasks as successes.
            finalized = float(
                stats.get(
                    "finalized",
                    float(stats.get("completed", 0)) + failures,
                )
            )
            failure_rate = (failures / finalized * 100.0) if finalized > 0 else 0.0
            # ===== RT_OPT BASELINE SAFETY CHANGE END =====

            old_w_load = int(
                round(
                    clamp(float(stats.get("curr_w_load", 0.5)), 0.10, 0.90)
                    * 100.0
                )
            )
            old_threshold = int(
                round(
                    clamp(
                        float(stats.get("curr_thr", 50.0)),
                        THRESHOLD_MIN,
                        THRESHOLD_MAX,
                    )
                )
            )
            # ===== RT-OPT ABLATION LOGGING CHANGE START =====
            old_w_load_values[app_id] = old_w_load / 100.0
            old_threshold_values[app_id] = float(old_threshold)
            # ===== RT-OPT ABLATION LOGGING CHANGE END =====

            # ===== RT_OPT BASELINE SAFETY CHANGE START: same feasible domain as SCOPE =====
            w_load = model.NewIntVar(
                W_LOAD_MIN_PERCENT,
                W_LOAD_MAX_PERCENT,
                f"wl_{app_id}",
            )
            w_dist = model.NewIntVar(
                W_LOAD_MIN_PERCENT,
                W_LOAD_MAX_PERCENT,
                f"wd_{app_id}",
            )
            threshold = model.NewIntVar(
                THRESHOLD_MIN,
                THRESHOLD_MAX,
                f"th_{app_id}",
            )
            model.Add(w_load + w_dist == 100)
            # ===== RT_OPT BASELINE SAFETY CHANGE END =====

            # ===== RT_OPT BASELINE SAFETY CHANGE START: bounded constraints =====
            # Every dynamically constructed bound is clamped to the variable's
            # legal domain, preventing infeasible models at 20/95 or 10/90.
            if failure_rate > 15.0:
                model.Add(threshold <= 40)
                model.Add(w_load >= 90)
            elif failure_rate > 5.0:
                threshold_upper = max(THRESHOLD_MIN, old_threshold - 10)
                w_load_lower = min(W_LOAD_MAX_PERCENT, old_w_load + 10)
                model.Add(threshold <= threshold_upper)
                model.Add(w_load >= w_load_lower)
            else:
                threshold_lower = max(THRESHOLD_MIN, old_threshold - 2)
                threshold_upper = min(THRESHOLD_MAX, old_threshold + 2)
                model.Add(threshold >= threshold_lower)
                model.Add(threshold <= threshold_upper)
            # ===== RT_OPT BASELINE SAFETY CHANGE END =====

            diff_w_load = model.NewIntVar(0, 100, f"diff_w_{app_id}")
            diff_threshold = model.NewIntVar(0, 100, f"diff_t_{app_id}")
            model.AddAbsEquality(diff_w_load, w_load - old_w_load)
            model.AddAbsEquality(diff_threshold, threshold - old_threshold)
            churn_terms.append(diff_w_load + diff_threshold)
            variables[app_id] = (w_load, w_dist, threshold)

        model.Minimize(sum(churn_terms))
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            sections = [f"ALPHA:{new_alpha:.2f}"]
            # ===== RT-OPT ABLATION LOGGING CHANGE START =====
            selected_w_load_values = {}
            selected_threshold_values = {}
            # ===== RT-OPT ABLATION LOGGING CHANGE END =====
            for app_id, (w_load, w_dist, threshold) in variables.items():
                # ===== RT-OPT ABLATION LOGGING CHANGE START =====
                selected_w_load_values[app_id] = solver.Value(w_load) / 100.0
                selected_threshold_values[app_id] = float(solver.Value(threshold))
                # ===== RT-OPT ABLATION LOGGING CHANGE END =====
                sections.append(
                    f"{app_id}:{solver.Value(w_load) / 100.0:.2f}:"
                    f"{solver.Value(w_dist) / 100.0:.2f}:"
                    f"{float(solver.Value(threshold)):.1f}"
                )
            # ===== RT-OPT ABLATION LOGGING CHANGE START =====
            if set(variables) != set(range(APP_COUNT)):
                raise ValueError(
                    "RT_OPT expected application IDs 0, 1, and 2 for "
                    "comparable normalized churn"
                )
            reconfiguration_cost = _normalized_reconfiguration_cost(
                curr_alpha,
                new_alpha,
                old_w_load_values,
                selected_w_load_values,
                old_threshold_values,
                selected_threshold_values,
            )
            optimizer_time_ms = (time.perf_counter() - optimizer_start) * 1000.0
            _write_decision_log(
                data,
                curr_alpha,
                new_alpha,
                old_w_load_values,
                selected_w_load_values,
                old_threshold_values,
                selected_threshold_values,
                reconfiguration_cost,
                optimizer_time_ms,
                "feasible",
            )
            # ===== RT-OPT ABLATION LOGGING CHANGE END =====
            return "|".join(sections) + "|"

        logging.error("RT_OPT found no feasible solution; retaining current alpha.")
        # ===== RT-OPT ABLATION LOGGING CHANGE START =====
        if set(old_w_load_values) == set(range(APP_COUNT)):
            optimizer_time_ms = (time.perf_counter() - optimizer_start) * 1000.0
            _write_decision_log(
                data,
                curr_alpha,
                curr_alpha,
                old_w_load_values,
                old_w_load_values,
                old_threshold_values,
                old_threshold_values,
                0.0,
                optimizer_time_ms,
                "infeasible_retain_previous",
            )
        # ===== RT-OPT ABLATION LOGGING CHANGE END =====
        return f"ALPHA:{curr_alpha:.2f}|"

    except Exception as exc:
        logging.exception("RT_OPT failed: %s", exc)
        return "ERROR", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
