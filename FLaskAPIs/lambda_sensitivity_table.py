#!/usr/bin/env python3
"""Create journal-ready SCOPE lambda-sensitivity tables.

Example
-------
python lambda_sensitivity_table.py \
    --decision-input "Lambda testing_Flask.zip" \
    --results-input "Lambda testing_Simresults.zip" \
    --output-dir lambda_table_output \
    --selected-lambda 0.20

The script accepts either ZIP archives or already-extracted directories. It
creates an auditable numeric CSV, a high-resolution PNG table, a vector PDF,
and a LaTeX table. Reported values are means plus/minus sample standard
deviations across paired iterations.
"""

# ===== JOURNAL TABLE GENERATOR STARTS HERE =====

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import statistics
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


DECISION_NAME = re.compile(
    r"lambda(?P<lambda>\d+)_rmax(?P<rmax>\d+)_ite(?P<iteration>\d+)\.csv$"
)
RESULT_PATH = re.compile(
    r"lambda(?P<lambda>\d+)[/\\]ite(?P<iteration>\d+)[/\\]"
)
EPSILON = 1e-9


@dataclass(frozen=True)
class DecisionRun:
    lambda_value: float
    iteration: int
    r_max: float
    decision_count: int
    reconfigurations: int
    total_churn: float
    max_step_churn: float
    constraint_active: int
    bound_violations: int


@dataclass(frozen=True)
class PerformanceRun:
    lambda_value: float
    iteration: int
    completed_tasks: int
    failed_tasks: int
    uncompleted_tasks: int
    failure_rate_pct: float
    service_time_s: float
    qoe_completed_pct: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate SCOPE lambda tests and draw a journal table."
    )
    parser.add_argument(
        "--decision-input",
        type=Path,
        required=True,
        help="Decision-log ZIP archive or extracted directory.",
    )
    parser.add_argument(
        "--results-input",
        type=Path,
        required=True,
        help="EdgeCloudSim result ZIP archive or extracted directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lambda_table_output"),
        help="Directory for CSV, PNG, PDF, and LaTeX outputs.",
    )
    parser.add_argument(
        "--selected-lambda",
        type=float,
        default=0.20,
        help="Lambda row to emphasize in the journal table (default: 0.20).",
    )
    return parser.parse_args()


def materialize_input(source: Path, destination: Path) -> Path:
    """Return a directory containing source, extracting ZIP files when needed."""
    source = source.expanduser().resolve()
    if source.is_dir():
        return source
    if not source.is_file():
        raise FileNotFoundError(f"Input does not exist: {source}")
    if not zipfile.is_zipfile(source):
        raise ValueError(f"Expected a ZIP archive or directory: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        archive.extractall(destination)
    return destination


def is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def read_decision_runs(root: Path) -> dict[tuple[float, int], DecisionRun]:
    runs: dict[tuple[float, int], DecisionRun] = {}
    for path in sorted(root.rglob("lambda*_rmax*_ite*.csv")):
        if path.name.startswith("._") or "__MACOSX" in path.parts:
            continue
        match = DECISION_NAME.fullmatch(path.name)
        if not match:
            continue

        lambda_from_name = int(match.group("lambda")) / 100.0
        rmax_from_name = int(match.group("rmax")) / 100.0
        iteration = int(match.group("iteration"))

        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"Decision log is empty: {path}")

        logged_lambdas = {float(row["lambda"]) for row in rows}
        logged_rmax = {float(row["r_max"]) for row in rows}
        hard_flags = {is_true(row["hard_stability_enabled"]) for row in rows}
        if logged_lambdas != {lambda_from_name}:
            raise ValueError(f"Filename/logged lambda mismatch in {path}")
        if logged_rmax != {rmax_from_name}:
            raise ValueError(f"Filename/logged R_max mismatch in {path}")
        if hard_flags != {True}:
            raise ValueError(f"Hard stability was not enabled throughout {path}")

        churn = [float(row["reconfiguration_cost"]) for row in rows]
        run = DecisionRun(
            lambda_value=lambda_from_name,
            iteration=iteration,
            r_max=rmax_from_name,
            decision_count=len(rows),
            reconfigurations=sum(value > EPSILON for value in churn),
            total_churn=sum(churn),
            max_step_churn=max(churn),
            constraint_active=sum(is_true(row["constraint_active"]) for row in rows),
            bound_violations=sum(value > rmax_from_name + EPSILON for value in churn),
        )
        key = (lambda_from_name, iteration)
        if key in runs:
            raise ValueError(f"Duplicate decision run for lambda/iteration {key}")
        runs[key] = run

    if not runs:
        raise ValueError(f"No decision CSV files were found under {root}")
    return runs


def meaningful_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_performance_runs(root: Path) -> dict[tuple[float, int], PerformanceRun]:
    runs: dict[tuple[float, int], PerformanceRun] = {}
    for path in sorted(root.rglob("*_ALL_APPS_GENERIC.log")):
        if "__MACOSX" in path.parts:
            continue
        match = RESULT_PATH.search(str(path))
        if not match:
            continue

        lambda_value = int(match.group("lambda")) / 100.0
        iteration = int(match.group("iteration"))
        lines = meaningful_lines(path)
        if len(lines) < 2:
            raise ValueError(f"Malformed generic result log: {path}")
        values = lines[1].split(";")
        if len(values) < 13:
            raise ValueError(f"Expected at least 13 fields in {path}")

        completed = int(values[0])
        failed = int(values[1])
        uncompleted = int(values[2])
        finalized = completed + failed
        failure_rate = 100.0 * failed / finalized if finalized else math.nan

        run = PerformanceRun(
            lambda_value=lambda_value,
            iteration=iteration,
            completed_tasks=completed,
            failed_tasks=failed,
            uncompleted_tasks=uncompleted,
            failure_rate_pct=failure_rate,
            service_time_s=float(values[4]),
            qoe_completed_pct=float(values[11]),
        )
        key = (lambda_value, iteration)
        if key in runs:
            raise ValueError(f"Duplicate performance run for lambda/iteration {key}")
        runs[key] = run

    if not runs:
        raise ValueError(f"No ALL_APPS_GENERIC logs were found under {root}")
    return runs


def validate_pairing(
    decisions: dict[tuple[float, int], DecisionRun],
    performance: dict[tuple[float, int], PerformanceRun],
    results_root: Path,
) -> tuple[list[float], list[int], float, int]:
    if set(decisions) != set(performance):
        only_decisions = sorted(set(decisions) - set(performance))
        only_performance = sorted(set(performance) - set(decisions))
        raise ValueError(
            "Decision/result run mismatch. "
            f"Only decisions={only_decisions}; only results={only_performance}"
        )

    lambdas = sorted({key[0] for key in decisions})
    iterations = sorted({key[1] for key in decisions})
    expected = {(value, iteration) for value in lambdas for iteration in iterations}
    if set(decisions) != expected:
        raise ValueError("The lambda-by-iteration experiment grid is incomplete")

    rmax_values = {run.r_max for run in decisions.values()}
    decision_counts = {run.decision_count for run in decisions.values()}
    violations = sum(run.bound_violations for run in decisions.values())
    if len(rmax_values) != 1:
        raise ValueError(f"Multiple R_max values detected: {sorted(rmax_values)}")
    if len(decision_counts) != 1:
        raise ValueError(f"Runs have different decision counts: {sorted(decision_counts)}")
    if violations:
        raise ValueError(f"Detected {violations} hard-stability bound violations")

    # Verify paired mobility traces when LOCATION logs are available.
    for iteration in iterations:
        hashes: set[str] = set()
        for lambda_value in lambdas:
            lambda_tag = int(round(lambda_value * 100))
            candidates = list(
                results_root.rglob(
                    f"lambda{lambda_tag:03d}/ite{iteration}/*_LOCATION.log"
                )
            )
            candidates = [path for path in candidates if "__MACOSX" not in path.parts]
            if len(candidates) == 1:
                hashes.add(hashlib.sha256(candidates[0].read_bytes()).hexdigest())
        if hashes and len(hashes) != 1:
            raise ValueError(
                f"Iteration {iteration} does not use the same mobility trace across lambdas"
            )

    return lambdas, iterations, next(iter(rmax_values)), next(iter(decision_counts))


def sample_sd(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_runs(
    lambdas: list[float],
    iterations: list[int],
    decisions: dict[tuple[float, int], DecisionRun],
    performance: dict[tuple[float, int], PerformanceRun],
) -> list[dict[str, float]]:
    summary: list[dict[str, float]] = []
    metrics = {
        "failure_rate_pct": lambda d, p: p.failure_rate_pct,
        "service_time_s": lambda d, p: p.service_time_s,
        "qoe_completed_pct": lambda d, p: p.qoe_completed_pct,
        "completed_tasks": lambda d, p: float(p.completed_tasks),
        "reconfigurations": lambda d, p: float(d.reconfigurations),
        "total_churn": lambda d, p: d.total_churn,
        "max_step_churn": lambda d, p: d.max_step_churn,
        "constraint_active": lambda d, p: float(d.constraint_active),
    }

    for lambda_value in lambdas:
        row: dict[str, float] = {
            "lambda": lambda_value,
            "runs": float(len(iterations)),
        }
        for metric_name, getter in metrics.items():
            values = [
                getter(
                    decisions[(lambda_value, iteration)],
                    performance[(lambda_value, iteration)],
                )
                for iteration in iterations
            ]
            row[f"{metric_name}_mean"] = statistics.mean(values)
            row[f"{metric_name}_sd"] = sample_sd(values)
        summary.append(row)
    return summary


def write_summary_csv(summary: list[dict[str, float]], output_path: Path) -> None:
    fieldnames = list(summary[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def mean_sd(row: dict[str, float], metric: str, decimals: int) -> str:
    return (
        f"{row[f'{metric}_mean']:.{decimals}f} "
        f"\u00b1 {row[f'{metric}_sd']:.{decimals}f}"
    )


def table_rows(summary: list[dict[str, float]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in summary:
        rows.append(
            [
                f"{row['lambda']:.2f}",
                mean_sd(row, "failure_rate_pct", 3),
                mean_sd(row, "service_time_s", 4),
                mean_sd(row, "qoe_completed_pct", 3),
                mean_sd(row, "reconfigurations", 2),
                mean_sd(row, "total_churn", 4),
                mean_sd(row, "max_step_churn", 4),
                mean_sd(row, "constraint_active", 2),
            ]
        )
    return rows


def draw_table(
    summary: list[dict[str, float]],
    selected_lambda: float,
    r_max: float,
    decision_count: int,
    png_path: Path,
    pdf_path: Path,
) -> None:
    headers = [
        r"$\lambda$",
        "Failure rate\n(%)",
        "Service time\n(s)",
        "QoE\n(%)",
        "Reconfigurations\nper run",
        "Total churn\nper run",
        r"Max. $R_t$" + "\nper run",
        f"Constraint-active\ndecisions (/{decision_count})",
    ]
    rows = table_rows(summary)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
        }
    )
    figure, axis = plt.subplots(figsize=(16.0, 4.8))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.12, 0.98, 0.76],
        colWidths=[0.06, 0.15, 0.15, 0.14, 0.14, 0.13, 0.11, 0.17],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.55)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.65)
        if row_index == 0:
            cell.set_facecolor("#303030")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            lambda_value = summary[row_index - 1]["lambda"]
            if math.isclose(lambda_value, selected_lambda, abs_tol=1e-12):
                cell.set_facecolor("#D9D9D9")
                cell.get_text().set_weight("bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#F5F5F5")
            else:
                cell.set_facecolor("white")

    figure.suptitle(
        rf"SCOPE sensitivity to $\lambda$ ($R_{{\max}}={r_max:.2f}$, "
        rf"$n={int(summary[0]['runs'])}$ paired runs)",
        fontsize=13,
        fontweight="bold",
        y=0.96,
    )
    figure.text(
        0.5,
        0.045,
        "Values are mean \u00b1 sample standard deviation. "
        f"The shaded row marks the selected λ = {selected_lambda:.2f}.",
        ha="center",
        va="center",
        fontsize=9,
    )
    figure.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def latex_mean_sd(
    row: dict[str, float], metric: str, decimals: int, bold: bool = False
) -> str:
    expression = (
        f"{row[f'{metric}_mean']:.{decimals}f} "
        f"\\pm {row[f'{metric}_sd']:.{decimals}f}"
    )
    return f"$\\mathbf{{{expression}}}$" if bold else f"${expression}$"


def write_latex_table(
    summary: list[dict[str, float]],
    selected_lambda: float,
    r_max: float,
    decision_count: int,
    output_path: Path,
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Sensitivity of SCOPE to $\lambda$ with "
            rf"$R_{{\max}}={r_max:.2f}$. Values are mean $\pm$ sample standard "
            rf"deviation over {int(summary[0]['runs'])} paired runs.}}"
        ),
        r"\label{tab:scope-lambda-sensitivity}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{c c c c c c c c}",
        r"\toprule",
        (
            r"$\lambda$ & Failure rate (\%) & Service time (s) & QoE (\%) & "
            r"Reconfigurations & Total churn & Max. $R_t$ & "
            rf"Constraint active (/{decision_count}) \\"
        ),
        r"\midrule",
    ]
    for row in summary:
        selected = math.isclose(
            row["lambda"], selected_lambda, abs_tol=1e-12
        )
        values = [
            (
                rf"\textbf{{{row['lambda']:.2f}}}"
                if selected
                else f"{row['lambda']:.2f}"
            ),
            latex_mean_sd(row, "failure_rate_pct", 3, selected),
            latex_mean_sd(row, "service_time_s", 4, selected),
            latex_mean_sd(row, "qoe_completed_pct", 3, selected),
            latex_mean_sd(row, "reconfigurations", 2, selected),
            latex_mean_sd(row, "total_churn", 4, selected),
            latex_mean_sd(row, "max_step_churn", 4, selected),
            latex_mean_sd(row, "constraint_active", 2, selected),
        ]
        line = " & ".join(values) + r" \\"
        lines.append(line)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="scope_lambda_") as temporary:
        temporary_root = Path(temporary)
        decisions_root = materialize_input(
            args.decision_input, temporary_root / "decisions"
        )
        results_root = materialize_input(
            args.results_input, temporary_root / "results"
        )

        decisions = read_decision_runs(decisions_root)
        performance = read_performance_runs(results_root)
        lambdas, iterations, r_max, decision_count = validate_pairing(
            decisions, performance, results_root
        )
        if not any(
            math.isclose(value, args.selected_lambda, abs_tol=1e-12)
            for value in lambdas
        ):
            raise ValueError(
                f"Selected lambda {args.selected_lambda} is absent; found {lambdas}"
            )

        summary = aggregate_runs(lambdas, iterations, decisions, performance)

    csv_path = args.output_dir / "lambda_sensitivity_summary.csv"
    png_path = args.output_dir / "lambda_sensitivity_table.png"
    pdf_path = args.output_dir / "lambda_sensitivity_table.pdf"
    latex_path = args.output_dir / "lambda_sensitivity_table.tex"

    write_summary_csv(summary, csv_path)
    draw_table(
        summary,
        args.selected_lambda,
        r_max,
        decision_count,
        png_path,
        pdf_path,
    )
    write_latex_table(
        summary,
        args.selected_lambda,
        r_max,
        decision_count,
        latex_path,
    )

    print(f"Validated {len(lambdas)} lambdas x {len(iterations)} paired runs.")
    print(f"R_max={r_max:.2f}; decisions/run={decision_count}; bound violations=0.")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {latex_path}")


if __name__ == "__main__":
    main()

# ===== JOURNAL TABLE GENERATOR ENDS HERE =====
