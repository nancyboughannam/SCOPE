#!/usr/bin/env python3
"""Create journal-ready SCOPE ablation figures and summary tables.

The script combines EdgeCloudSim ALL_APPS_GENERIC logs with optimizer decision
CSVs. Statistical uncertainty is calculated across independent paired runs;
individual optimizer decisions are used only for descriptive churn ECDFs.
"""

# ===== JOURNAL ABLATION ANALYSIS CHANGE START =====

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


VARIANT_ORDER = [
    "RT_OPT_ML",
    "SOFTCOST_OPT",
    "PERFORMANCE_ONLY",
    "RT_OPT",
]
DISPLAY_NAMES = {
    "RT_OPT_ML": "SCOPE",
    "SOFTCOST_OPT": "SoftCost-OPT",
    "PERFORMANCE_ONLY": "Performance-only",
    "RT_OPT": "RT-OPT",
}
POLICY_TO_VARIANT = {
    # ===== FOUR REQUIRED RAW POLICIES CHANGE START =====
    # RT_OPT_ML is the simulator's raw identifier for the revised SCOPE
    # formulation; the journal display name remains SCOPE.
    "SCOPE": "RT_OPT_ML",
    "RT_OPT_ML": "RT_OPT_ML",
    "SOFTCOST_OPT": "SOFTCOST_OPT",
    "PERFORMANCE_ONLY": "PERFORMANCE_ONLY",
    "RT_OPT": "RT_OPT",
    # ===== FOUR REQUIRED RAW POLICIES CHANGE END =====
}
COLORS = {
    "RT_OPT_ML": "#0072B2",
    "SOFTCOST_OPT": "#E69F00",
    "PERFORMANCE_ONLY": "#CC79A7",
    "RT_OPT": "#666666",
}
HATCHES = {
    "RT_OPT_ML": "",
    "SOFTCOST_OPT": "//",
    "PERFORMANCE_ONLY": "xx",
    "RT_OPT": "..",
}


# ===== HARD-CODED ABLATION LOCATIONS CHANGE START =====
# The script can now be started directly with:
#     python plot_ablation_results.py
#
# Simulator files are stored in ite1, ite2, ... below this directory.
SIM_RESULTS_DIR = Path(
    "/Users/nancyboughannam/Documents/ThreeBrains/sim_results"
)

# This is the parent directory that contains decision_logs.
DECISION_RESULTS_DIR = Path(
    "/Users/nancyboughannam/PycharmProjects/FLaskAPIs/ablation_results_final"
)

OUTPUT_DIR = Path(
    "/Users/nancyboughannam/PycharmProjects/FLaskAPIs/ablation_figures_final"
)

VEHICLES = 1800
EXPECTED_ITERATIONS = 10  # Change only this number to 4, 5, ..., then 10.
BASE_SEED = 20260828
R_MAX = 0.15
CHURN_TOLERANCE = 1e-12

# IMPORTANT RT-OPT FILENAME DIFFERENCE:
# The three ML-based policies may currently have short filenames such as:
#     scope_ite1.csv
#     softcost_ite1.csv
#     performance_only_ite1.csv
# RT-OPT uses a different, longer automatic filename such as:
#     rt_opt_iteration1_seed20260829_n1800_iteration_1.csv
# load_decisions() scans every CSV recursively and uses its `variant` column,
# so both filename styles are intentionally supported.
# ===== HARD-CODED ABLATION LOCATIONS CHANGE END =====


def _iteration_from_text(value: object) -> int | None:
    match = re.search(r"(?:ite|iteration)[_-]?(\d+)", str(value), re.I)
    return int(match.group(1)) if match else None


def _iteration_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        iteration = _iteration_from_text(part)
        if iteration is not None:
            return iteration
    return None


def _generic_identity(path: Path) -> tuple[str, int] | None:
    policies = "|".join(sorted(POLICY_TO_VARIANT, key=len, reverse=True))
    pattern = re.compile(
        rf"SIMRESULT_.+_(?P<policy>{policies})_"
        rf"(?P<devices>\d+)DEVICES_ALL_APPS_GENERIC\.log$"
    )
    match = pattern.match(path.name)
    if not match:
        return None
    return match.group("policy"), int(match.group("devices"))


def _first_numeric_row(path: Path) -> list[float]:
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            return [float(value) for value in line.split(";")]
        except ValueError as exc:
            raise ValueError(f"Non-numeric generic-log row in {path}: {line}") from exc
    raise ValueError(f"No numeric rows found in generic log: {path}")


def load_performance_runs(
    results_root: Path,
    vehicles: int,
    base_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(results_root.rglob("*ALL_APPS_GENERIC.log")):
        identity = _generic_identity(path)
        if identity is None:
            continue
        policy, device_count = identity
        if device_count != vehicles:
            continue
        iteration = _iteration_from_path(path)
        if iteration is None:
            raise ValueError(f"Cannot identify iteN directory for {path}")

        values = _first_numeric_row(path)
        if len(values) < 5:
            raise ValueError(f"Expected at least five columns in {path}")
        completed, failed, uncompleted = values[0], values[1], values[2]
        finalized = completed + failed
        if finalized <= 0:
            raise ValueError(f"No finalized tasks in {path}")

        rows.append(
            {
                "variant": POLICY_TO_VARIANT[policy],
                "raw_policy": policy,
                "iteration": iteration,
                "simulation_seed": base_seed + iteration,
                "num_devices": device_count,
                "completed_tasks": completed,
                "failed_tasks": failed,
                "uncompleted_tasks": uncompleted,
                "generated_tasks": finalized + uncompleted,
                "finalized_tasks": finalized,
                "failure_rate_percent": 100.0 * failed / finalized,
                "average_service_time_seconds": values[4],
                "generic_log": str(path),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(
            f"No {vehicles}-vehicle ALL_APPS_GENERIC logs found under {results_root}"
        )
    duplicate = frame.duplicated(["variant", "iteration", "num_devices"], keep=False)
    if duplicate.any():
        details = frame.loc[duplicate, ["variant", "iteration", "generic_log"]]
        raise ValueError(f"Duplicate performance runs detected:\n{details.to_string(index=False)}")
    return frame


def _coerce_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "on"})


def _infer_variant(frame: pd.DataFrame, path: Path) -> pd.Series:
    if "variant" in frame.columns:
        raw = frame["variant"].astype(str).str.upper()
        return raw.map(lambda value: POLICY_TO_VARIANT.get(value, value))
    name = path.name.upper()
    for variant in VARIANT_ORDER:
        if variant in name:
            return pd.Series([variant] * len(frame), index=frame.index)
    if "hard_stability_enabled" in frame.columns:
        hard = _coerce_bool(frame["hard_stability_enabled"])
        lambda_values = pd.to_numeric(frame.get("lambda", 0.0), errors="coerce").fillna(0.0)
        inferred = np.where(
            hard,
            "RT_OPT_ML",
            np.where(lambda_values <= 1e-12, "PERFORMANCE_ONLY", "SOFTCOST_OPT"),
        )
        return pd.Series(inferred, index=frame.index)
    raise ValueError(f"Cannot infer variant for decision log {path}")


def load_decisions(decision_root: Path, base_seed: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    # ===== RT-OPT DIFFERENT FILENAME SUPPORT CHANGE START =====
    # Do not use a policy-specific filename pattern here. In particular,
    # RT-OPT's filename is rt_opt_iterationN_seed... rather than rt_opt_iteN.
    # Recursive CSV discovery lets the `variant` column identify every policy.
    for path in sorted(decision_root.rglob("*.csv")):
        # ===== RT-OPT DIFFERENT FILENAME SUPPORT CHANGE END =====
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "reconfiguration_cost" not in frame.columns or "timestamp" not in frame.columns:
            continue
        frame = frame.copy()
        frame["variant"] = _infer_variant(frame, path)
        if "iteration" not in frame.columns:
            frame["iteration"] = np.nan
        frame["iteration"] = pd.to_numeric(frame["iteration"], errors="coerce")

        missing_iteration = frame["iteration"].isna()
        if missing_iteration.any():
            fallback = None
            if "run_id" in frame.columns and not frame["run_id"].dropna().empty:
                fallback = _iteration_from_text(frame["run_id"].dropna().iloc[0])
            if fallback is None:
                fallback = _iteration_from_path(path)
            if fallback is not None:
                frame.loc[missing_iteration, "iteration"] = fallback

        if frame["iteration"].isna().any():
            raise ValueError(f"Decision log lacks iteration identity: {path}")
        frame["iteration"] = frame["iteration"].astype(int)

        if "simulation_seed" not in frame.columns:
            frame["simulation_seed"] = frame["iteration"] + base_seed
        frame["simulation_seed"] = pd.to_numeric(
            frame["simulation_seed"], errors="coerce"
        ).fillna(frame["iteration"] + base_seed).astype(int)

        frame["num_devices"] = pd.to_numeric(frame["num_devices"], errors="raise").astype(int)
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise")
        frame["reconfiguration_cost"] = pd.to_numeric(
            frame["reconfiguration_cost"], errors="raise"
        )
        if "optimizer_time_ms" in frame.columns:
            frame["optimizer_time_ms"] = pd.to_numeric(
                frame["optimizer_time_ms"], errors="coerce"
            )
        else:
            frame["optimizer_time_ms"] = np.nan
        frame["source_csv"] = str(path)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No decision CSV containing reconfiguration_cost found under {decision_root}"
        )
    decisions = pd.concat(frames, ignore_index=True, sort=False)
    decisions = decisions.sort_values(
        ["variant", "iteration", "num_devices", "timestamp"]
    ).reset_index(drop=True)

    # ===== DUPLICATE DECISION GUARD CHANGE START =====
    # A repeated variant/iteration/timestamp normally means that the same
    # iteration was rerun into an existing output tree. Failing here prevents
    # two executions from being silently treated as one independent run.
    duplicate_keys = ["variant", "iteration", "num_devices", "timestamp"]
    duplicate = decisions.duplicated(duplicate_keys, keep=False)
    if duplicate.any():
        details = decisions.loc[duplicate, duplicate_keys + ["source_csv"]]
        raise ValueError(
            "Duplicate optimizer decisions detected; keep only one execution "
            "per variant and iteration:\n" + details.to_string(index=False)
        )
    # ===== DUPLICATE DECISION GUARD CHANGE END =====
    return decisions


def summarize_decision_runs(
    decisions: pd.DataFrame,
    vehicles: int,
    r_max: float,
    churn_tolerance: float,
) -> pd.DataFrame:
    decisions = decisions.loc[decisions["num_devices"] == vehicles].copy()
    if decisions.empty:
        raise ValueError(f"Decision logs contain no rows for {vehicles} vehicles")
    decisions["reconfigured_derived"] = decisions["reconfiguration_cost"] > churn_tolerance
    decisions["violation_derived"] = decisions["reconfiguration_cost"] > r_max + 1e-12

    rows: list[dict[str, object]] = []
    keys = ["variant", "iteration", "simulation_seed", "num_devices"]
    for key, group in decisions.groupby(keys, sort=True):
        variant, iteration, seed, devices = key
        rows.append(
            {
                "variant": variant,
                "iteration": int(iteration),
                "simulation_seed": int(seed),
                "num_devices": int(devices),
                "optimizer_decisions": len(group),
                "reconfiguration_count": int(group["reconfigured_derived"].sum()),
                "mean_churn": group["reconfiguration_cost"].mean(),
                "max_churn": group["reconfiguration_cost"].max(),
                "cumulative_churn": group["reconfiguration_cost"].sum(),
                "violation_rate_percent": 100.0 * group["violation_derived"].mean(),
                "mean_optimizer_time_ms": group["optimizer_time_ms"].mean(),
            }
        )
    return pd.DataFrame(rows)


def validate_pairing(
    runs: pd.DataFrame,
    expected_iterations: int,
    base_seed: int,
) -> pd.DataFrame:
    required = set(VARIANT_ORDER)
    present = set(runs["variant"].unique())
    missing_variants = required - present
    # ===== REQUIRE ALL FOUR POLICIES CHANGE START =====
    # Even a one-iteration preview must contain every ablation policy. This
    # prevents --allow-incomplete from silently producing an RT-OPT-only plot.
    if missing_variants:
        raise ValueError(
            "Missing required policies: "
            f"{sorted(missing_variants)}. Run RT_OPT_ML, SOFTCOST_OPT, "
            "PERFORMANCE_ONLY, and RT_OPT in the same simulator invocation."
        )
    # ===== REQUIRE ALL FOUR POLICIES CHANGE END =====

    variants = [variant for variant in VARIANT_ORDER if variant in present]
    expected = set(range(1, expected_iterations + 1))
    iteration_sets = {
        variant: set(runs.loc[runs["variant"] == variant, "iteration"].astype(int))
        for variant in variants
    }
    for variant, observed in iteration_sets.items():
        if observed != expected:
            raise ValueError(
                f"{variant} iterations are {sorted(observed)}; expected {sorted(expected)}"
            )

    expected_seed = runs["iteration"].astype(int) + base_seed
    incorrect = runs["simulation_seed"].astype(int) != expected_seed
    if incorrect.any():
        details = runs.loc[incorrect, ["variant", "iteration", "simulation_seed"]]
        raise ValueError(f"Seed/iteration mismatch:\n{details.to_string(index=False)}")

    duplicate = runs.duplicated(["variant", "iteration"], keep=False)
    if duplicate.any():
        details = runs.loc[duplicate, ["variant", "iteration"]]
        raise ValueError(f"Duplicate joined runs:\n{details.to_string(index=False)}")

    # ===== PAIRED WORKLOAD CHECK CHANGE START =====
    # With a policy-independent seed, every policy must receive exactly the
    # same number of generated tasks within an iteration. This catches a seed
    # or logging regression before an unfair comparison reaches the paper.
    task_count_variants = runs.groupby("iteration")["generated_tasks"].nunique()
    mismatched_iterations = task_count_variants[task_count_variants > 1].index.tolist()
    if mismatched_iterations:
        details = runs.loc[
            runs["iteration"].isin(mismatched_iterations),
            ["variant", "iteration", "generated_tasks", "generic_log"],
        ]
        raise ValueError(
            "Generated-task counts differ between paired policies:\n"
            + details.to_string(index=False)
        )
    # ===== PAIRED WORKLOAD CHECK CHANGE END =====
    return runs


def mean_ci95(values: Iterable[float]) -> tuple[float, float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = len(array)
    if n == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(array.mean())
    if n == 1:
        return mean, mean, mean, 1
    half_width = float(student_t.ppf(0.975, n - 1) * array.std(ddof=1) / math.sqrt(n))
    return mean, mean - half_width, mean + half_width, n


def create_numeric_summary(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "completed_tasks",
        "failure_rate_percent",
        "average_service_time_seconds",
        "mean_churn",
        "max_churn",
        "cumulative_churn",
        "reconfiguration_count",
        "violation_rate_percent",
        "mean_optimizer_time_ms",
    ]
    rows: list[dict[str, object]] = []
    for variant in VARIANT_ORDER:
        group = runs.loc[runs["variant"] == variant]
        if group.empty:
            continue
        row: dict[str, object] = {
            "variant": variant,
            "display_name": DISPLAY_NAMES[variant],
        }
        for metric in metrics:
            mean, low, high, n = mean_ci95(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_n"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_performance_reconfiguration(runs: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("failure_rate_percent", "(a) Task failure rate", "Failure rate (%)"),
        ("average_service_time_seconds", "(b) Average service time", "Time (s)"),
        ("mean_churn", "(c) Mean normalized churn", "Mean $C_t$"),
        ("reconfiguration_count", "(d) Reconfiguration count", "Count per run"),
    ]
    variants = [v for v in VARIANT_ORDER if v in set(runs["variant"])]
    x = np.arange(len(variants))
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.9), constrained_layout=True)

    for axis, (metric, title, ylabel) in zip(axes.flat, metrics):
        means, errors = [], []
        for variant in variants:
            mean, low, high, _ = mean_ci95(runs.loc[runs["variant"] == variant, metric])
            means.append(mean)
            errors.append([[mean - low], [high - mean]])
        yerr = np.concatenate(errors, axis=1) if errors else None
        bars = axis.bar(
            x,
            means,
            yerr=yerr,
            capsize=3,
            color=[COLORS[v] for v in variants],
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, variant in zip(bars, variants):
            bar.set_hatch(HATCHES[variant])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [DISPLAY_NAMES[v] for v in variants], rotation=18, ha="right")
        axis.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0)

    # ===== EXACTLY TWO FIGURES AND ONE TABLE CHANGE START =====
    fig.savefig(
        output_dir / "figure1_performance_reconfiguration.pdf",
        bbox_inches="tight",
    )
    # ===== EXACTLY TWO FIGURES AND ONE TABLE CHANGE END =====
    plt.close(fig)


def plot_churn_ecdf(
    decisions: pd.DataFrame,
    vehicles: int,
    r_max: float,
    output_dir: Path,
) -> None:
    # ===== CLEAR NON-ZERO CHURN DISTRIBUTION CHANGE START =====
    # Figure 1(d) already reports how often each policy reconfigures. Plotting
    # every zero again in an ECDF made SCOPE and SoftCost-OPT jump to about 98%
    # at C_t=0 and overlap. Figure 2 therefore compares the magnitude of actual
    # reconfiguration events only (C_t > 0), using boxes plus all observations.
    frame = decisions.loc[decisions["num_devices"] == vehicles].copy()
    variants = [v for v in VARIANT_ORDER if v in set(frame["variant"])]
    distributions: list[np.ndarray] = []
    for variant in variants:
        values = frame.loc[
            (frame["variant"] == variant)
            & (frame["reconfiguration_cost"] > CHURN_TOLERANCE),
            "reconfiguration_cost",
        ].to_numpy(float)
        if len(values) == 0:
            raise ValueError(f"{DISPLAY_NAMES[variant]} has no non-zero churn events")
        distributions.append(values)

    positions = np.arange(1, len(variants) + 1)
    fig, axis = plt.subplots(figsize=(4.4, 3.05), constrained_layout=True)
    boxes = axis.boxplot(
        distributions,
        positions=positions,
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        boxprops={"color": "black", "linewidth": 0.8},
    )
    for patch, variant in zip(boxes["boxes"], variants):
        patch.set_facecolor(COLORS[variant])
        patch.set_alpha(0.28)
        patch.set_hatch(HATCHES[variant])

    markers = ["o", "s", "^", "D"]
    for index, (variant, values) in enumerate(zip(variants, distributions)):
        rng = np.random.default_rng(BASE_SEED + index)
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        axis.scatter(
            positions[index] + jitter,
            values,
            s=15,
            marker=markers[index],
            color=COLORS[variant],
            edgecolors="black",
            linewidths=0.25,
            alpha=0.72 if len(values) <= 10 else 0.32,
            zorder=3,
        )

    axis.axhline(
        r_max,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=rf"$R_{{\max}}={r_max:.2f}$",
    )
    axis.set_title("Reconfiguration magnitude when a change occurs")
    axis.set_ylabel("Normalized reconfiguration cost $C_t$")
    axis.set_xticks(
        positions,
        [DISPLAY_NAMES[variant] for variant in variants],
        rotation=16,
        ha="right",
    )
    axis.set_xlim(0.45, len(variants) + 0.55)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=True)
    # ===== EXACTLY TWO FIGURES AND ONE TABLE CHANGE START =====
    fig.savefig(
        output_dir / "figure2_churn_distribution.pdf",
        bbox_inches="tight",
    )
    # ===== EXACTLY TWO FIGURES AND ONE TABLE CHANGE END =====
    plt.close(fig)
    # ===== CLEAR NON-ZERO CHURN DISTRIBUTION CHANGE END =====


def _format_ci(mean: float, low: float, high: float, digits: int) -> str:
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def write_latex_table(summary: pd.DataFrame, output_dir: Path) -> list[list[str]]:
    columns = [
        ("completed_tasks", "Completed tasks", 0),
        ("failure_rate_percent", "Failure (\\%)", 2),
        ("average_service_time_seconds", "Service time (s)", 3),
        ("mean_churn", "Mean churn", 3),
        ("max_churn", "Max. churn", 3),
        ("reconfiguration_count", "Reconfigs.", 1),
        ("violation_rate_percent", "Violations (\\%)", 2),
    ]
    display_rows: list[list[str]] = []
    for _, row in summary.iterrows():
        cells = [str(row["display_name"])]
        for metric, _, digits in columns:
            cells.append(
                _format_ci(
                    float(row[f"{metric}_mean"]),
                    float(row[f"{metric}_ci_low"]),
                    float(row[f"{metric}_ci_high"]),
                    digits,
                )
            )
        display_rows.append(cells)

    header = "Method & " + " & ".join(label for _, label, _ in columns) + r" \\"
    body = "\n".join(" & ".join(row) + r" \\" for row in display_rows)
    latex = (
        "% ===== JOURNAL ABLATION TABLE CHANGE START =====\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Ablation results at 1,800 vehicles. Values are means "
        "with two-sided 95\\% confidence intervals across paired seeds.}\n"
        "\\label{tab:scope-ablation}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lccccccc}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table*}\n"
        "% ===== JOURNAL ABLATION TABLE CHANGE END =====\n"
    )
    (output_dir / "table1_ablation.tex").write_text(latex, encoding="utf-8")
    return display_rows


def main() -> None:
    # ===== USE HARD-CODED LOCATIONS CHANGE START =====
    if EXPECTED_ITERATIONS < 1:
        raise ValueError("EXPECTED_ITERATIONS must be at least 1")
    if not 0.0 <= R_MAX <= 1.0:
        raise ValueError("R_MAX must be between 0 and 1")
    if not SIM_RESULTS_DIR.is_dir():
        raise FileNotFoundError(f"Simulator-results directory not found: {SIM_RESULTS_DIR}")
    if not DECISION_RESULTS_DIR.is_dir():
        raise FileNotFoundError(f"Decision-results directory not found: {DECISION_RESULTS_DIR}")

    performance = load_performance_runs(SIM_RESULTS_DIR, VEHICLES, BASE_SEED)
    decisions = load_decisions(DECISION_RESULTS_DIR, BASE_SEED)
    decision_runs = summarize_decision_runs(
        decisions,
        VEHICLES,
        R_MAX,
        CHURN_TOLERANCE,
    )
    runs = performance.merge(
        decision_runs,
        on=["variant", "iteration", "simulation_seed", "num_devices"],
        how="inner",
        validate="one_to_one",
    )
    runs = validate_pairing(
        runs,
        EXPECTED_ITERATIONS,
        BASE_SEED,
    )

    scope_violations = runs.loc[
        runs["variant"] == "RT_OPT_ML", "violation_rate_percent"
    ]
    if not scope_violations.empty and (scope_violations > 0.0).any():
        raise ValueError(
            "SCOPE contains C_t > R_max decisions; verify hard-constraint enforcement"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = create_numeric_summary(runs)

    configure_matplotlib()
    plot_performance_reconfiguration(runs, OUTPUT_DIR)
    plot_churn_ecdf(decisions, VEHICLES, R_MAX, OUTPUT_DIR)
    write_latex_table(summary, OUTPUT_DIR)

    print(f"Validated paired runs: {len(runs)}")
    print(f"Variants: {', '.join(runs['variant'].drop_duplicates())}")
    if EXPECTED_ITERATIONS == 1:
        print("Preview only: one iteration cannot provide a confidence interval.")
    print("Created exactly two figures and one table:")
    print("  figure1_performance_reconfiguration.pdf")
    print("  figure2_churn_distribution.pdf")
    print("  table1_ablation.tex")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    # ===== USE HARD-CODED LOCATIONS CHANGE END =====


if __name__ == "__main__":
    main()

# ===== JOURNAL ABLATION ANALYSIS CHANGE END =====
