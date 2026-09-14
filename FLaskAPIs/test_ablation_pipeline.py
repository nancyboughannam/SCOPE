#!/usr/bin/env python3
"""End-to-end smoke test for the journal ablation analysis pipeline."""

# ===== ABLATION PIPELINE TEST CHANGE START =====

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


VARIANTS = {
    "SCOPE": "SCOPE",
    "SOFTCOST_OPT": "SOFTCOST_OPT",
    "PERFORMANCE_ONLY": "PERFORMANCE_ONLY",
    "RT_OPT": "RT_OPT",
}


def _write_generic_log(path: Path, completed: int, failed: int, service: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{completed};{failed};10;0;{service};0.4;0.1\n",
        encoding="utf-8",
    )


def _write_decisions(
    path: Path,
    variant: str,
    iteration: int,
    churn_values: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "policy",
        "run_id",
        "iteration",
        "simulation_seed",
        "timestamp",
        "num_devices",
        "reconfiguration_cost",
        "optimizer_time_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for decision_index, churn in enumerate(churn_values, start=1):
            writer.writerow(
                {
                    "variant": variant,
                    "policy": variant,
                    "run_id": f"iteration_{iteration}",
                    "iteration": iteration,
                    "simulation_seed": 20260828 + iteration,
                    "timestamp": 240 * decision_index,
                    "num_devices": 1800,
                    "reconfiguration_cost": churn,
                    "optimizer_time_ms": 2.0 + decision_index / 10.0,
                }
            )


def main() -> None:
    script = Path(__file__).with_name("plot_ablation_results.py")
    with tempfile.TemporaryDirectory(prefix="scope_ablation_test_") as directory:
        root = Path(directory)
        sim_root = root / "sim_results"
        decision_root = root / "decision_results"
        output_root = root / "journal_outputs"

        for iteration in range(1, 4):
            for variant_index, (policy, variant) in enumerate(VARIANTS.items()):
                failed = 100_000 + 5_000 * variant_index + 50 * iteration
                finalized = 1_100_000 + 100 * iteration
                completed = finalized - failed
                service = 0.50 + 0.03 * variant_index + 0.002 * iteration
                generic_name = (
                    f"SIMRESULT_ITS_SCENARIO_{policy}_"
                    "1800DEVICES_ALL_APPS_GENERIC.log"
                )
                _write_generic_log(
                    sim_root / f"ite{iteration}" / generic_name,
                    completed,
                    failed,
                    service,
                )

                if variant == "SCOPE":
                    churn = [0.00, 0.08, 0.15]
                elif variant == "SOFTCOST_OPT":
                    churn = [0.00, 0.12, 0.24]
                elif variant == "PERFORMANCE_ONLY":
                    churn = [0.05, 0.20, 0.35]
                else:
                    churn = [0.00, 0.10, 0.18]
                _write_decisions(
                    decision_root
                    / variant.lower()
                    / f"{variant.lower()}_iteration{iteration}.csv",
                    variant,
                    iteration,
                    churn,
                )

        command = [
            sys.executable,
            str(script),
            "--sim-results",
            str(sim_root),
            "--decision-results",
            str(decision_root),
            "--output",
            str(output_root),
            "--expected-iterations",
            "3",
            "--r-max",
            "0.15",
        ]
        subprocess.run(command, check=True)

        expected_files = {
            "ablation_performance_reconfiguration.pdf",
            "ablation_performance_reconfiguration.png",
            "ablation_churn_ecdf.pdf",
            "ablation_churn_ecdf.png",
            "ablation_summary_numeric.csv",
            "ablation_summary_table.tex",
            "ablation_summary_table.pdf",
            "ablation_summary_table.png",
            "ablation_run_level_metrics.csv",
            "ablation_decision_level_metrics.csv",
        }
        missing = [name for name in expected_files if not (output_root / name).exists()]
        if missing:
            raise AssertionError(f"Missing analysis outputs: {missing}")

        print("Ablation pipeline smoke test passed.")


if __name__ == "__main__":
    main()

# ===== ABLATION PIPELINE TEST CHANGE END =====
