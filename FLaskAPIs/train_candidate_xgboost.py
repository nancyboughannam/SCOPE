#!/usr/bin/env python3
"""Train and evaluate candidate-aware SCOPE performance models.

The script trains one XGBoost regressor per application and target:

* task-failure rate
* deadline-normalized service time

It keeps entire simulation iterations separated, uses validation data for model
selection/early stopping, and touches the test iteration only for final reporting.
It also trains a state-only XGBoost ablation so that good results caused only by
workload/state features are not mistaken for candidate-configuration learning.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


SEED = 42
SPLITS = ("train", "validation", "test")
TARGETS = {
    "failure_rate": "target_failure_rate",
    "deadline_norm_service_time": "target_deadline_norm_service_time",
}
CANDIDATE_COLUMNS = [
    "candidate_alpha",
    "candidate_w_load_0",
    "candidate_w_load_1",
    "candidate_w_load_2",
    "candidate_threshold_0",
    "candidate_threshold_1",
    "candidate_threshold_2",
]
METADATA_COLUMNS = {
    "run_id",
    "policy",
    "window_id",
    "window_start",
    "window_end",
    "app_id",
}

# A small, declared search space. It is evaluated on validation MAE only.
PARAMETER_CANDIDATES = [
    {
        "max_depth": 2,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
    },
    {
        "max_depth": 3,
        "learning_rate": 0.04,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
    },
    {
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train candidate-aware XGBoost models from prepared SCOPE splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("scope_candidate_dataset"),
        help="Directory containing candidate_dataset_{train,validation,test}.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scope_candidate_xgboost_results"),
        help="Directory that will receive models, metrics, plots, and reports.",
    )
    return parser.parse_args()


def load_and_validate(data_dir: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    schema_path = data_dir / "candidate_feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing feature schema: {schema_path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_features = list(schema["feature_columns"])
    feature_columns = [c for c in schema_features if c not in METADATA_COLUMNS]

    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = data_dir / f"candidate_dataset_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing prepared split: {path}")
        frame = pd.read_csv(path)
        missing = set(feature_columns + list(TARGETS.values())) - set(frame.columns)
        if missing:
            raise ValueError(f"{split} is missing columns: {sorted(missing)}")
        if frame[feature_columns + list(TARGETS.values())].isna().any().any():
            raise ValueError(f"{split} contains missing numeric values")
        if not np.isfinite(frame[feature_columns + list(TARGETS.values())].to_numpy()).all():
            raise ValueError(f"{split} contains infinite numeric values")
        if not frame["target_failure_rate"].between(0.0, 1.0).all():
            raise ValueError(f"{split} contains failure rates outside [0, 1]")
        if (frame["target_deadline_norm_service_time"] < 0.0).any():
            raise ValueError(f"{split} contains negative normalized service time")
        frames[split] = frame

    run_sets = {split: set(frames[split]["run_id"].unique()) for split in SPLITS}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = run_sets[left] & run_sets[right]
        if overlap:
            raise ValueError(f"Run leakage between {left} and {right}: {sorted(overlap)}")

    train_apps = sorted(frames["train"]["app_id"].unique().tolist())
    for split in ("validation", "test"):
        if sorted(frames[split]["app_id"].unique().tolist()) != train_apps:
            raise ValueError(f"Application IDs differ between train and {split}")

    return frames, feature_columns


def clipped_prediction(target_key: str, prediction: np.ndarray) -> np.ndarray:
    if target_key == "failure_rate":
        return np.clip(prediction, 0.0, 1.0)
    return np.clip(prediction, 0.0, None)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_model(params: dict[str, Any]) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        n_estimators=2400,
        early_stopping_rounds=75,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=4,
        tree_method="hist",
        verbosity=0,
        **params,
    )


def select_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    target_column: str,
    target_key: str,
) -> tuple[XGBRegressor, dict[str, Any], float]:
    x_train = train[features]
    y_train = train[target_column].to_numpy()
    x_validation = validation[features]
    y_validation = validation[target_column].to_numpy()

    best_model: XGBRegressor | None = None
    best_params: dict[str, Any] | None = None
    best_mae = float("inf")
    for params in PARAMETER_CANDIDATES:
        model = make_model(params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_validation, y_validation)],
            verbose=False,
        )
        prediction = clipped_prediction(target_key, model.predict(x_validation))
        score = float(mean_absolute_error(y_validation, prediction))
        if score < best_mae:
            best_model = model
            best_params = dict(params)
            best_mae = score

    if best_model is None or best_params is None:
        raise RuntimeError("No model was selected")
    return best_model, best_params, best_mae


def append_metric_rows(
    rows: list[dict[str, Any]],
    frames: dict[str, pd.DataFrame],
    app_id: int,
    target_key: str,
    target_column: str,
    model_type: str,
    predictor: Any,
    features: list[str] | None,
) -> None:
    for split in SPLITS:
        app_frame = frames[split].loc[frames[split]["app_id"] == app_id]
        y_true = app_frame[target_column].to_numpy()
        if model_type == "training_mean":
            y_pred = np.full(len(app_frame), float(predictor))
        else:
            y_pred = clipped_prediction(target_key, predictor.predict(app_frame[features]))
        row = {
            "app_id": int(app_id),
            "target": target_key,
            "model_type": model_type,
            "split": split,
            "rows": int(len(app_frame)),
        }
        row.update(metrics(y_true, y_pred))
        rows.append(row)


def load_band(num_devices: pd.Series) -> pd.Categorical:
    return pd.cut(
        num_devices,
        bins=[0, 600, 1200, float("inf")],
        labels=["low_100_600", "medium_700_1200", "high_1300_1800"],
    )


def candidate_sensitivity(
    model: XGBRegressor,
    features: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    app_id: int,
    target_key: str,
) -> list[dict[str, Any]]:
    """Measure prediction changes while holding the observed state fixed.

    Candidate configurations are sampled as complete, observed tuples so the
    test does not invent invalid combinations. This is a model diagnostic, not
    a causal estimate of the simulator's true counterfactual response.
    """

    candidates = train[CANDIDATE_COLUMNS].drop_duplicates()
    if len(candidates) > 50:
        candidates = candidates.sample(n=50, random_state=SEED + app_id)

    states = test.loc[test["num_devices"] >= 1300]
    if len(states) > 30:
        states = states.sample(n=30, random_state=SEED + 100 + app_id)

    detail: list[dict[str, Any]] = []
    for _, state_row in states.iterrows():
        repeated = pd.DataFrame(
            np.repeat(
                state_row[features].astype(float).to_numpy()[None, :],
                len(candidates),
                axis=0,
            ),
            columns=features,
        )
        repeated.loc[:, CANDIDATE_COLUMNS] = candidates[CANDIDATE_COLUMNS].to_numpy()
        prediction = clipped_prediction(target_key, model.predict(repeated))
        detail.append(
            {
                "app_id": int(app_id),
                "target": target_key,
                "run_id": state_row["run_id"],
                "num_devices": int(state_row["num_devices"]),
                "window_id": int(state_row["window_id"]),
                "candidate_count": int(len(candidates)),
                "prediction_min": float(np.min(prediction)),
                "prediction_max": float(np.max(prediction)),
                "prediction_range": float(np.ptp(prediction)),
                "prediction_std": float(np.std(prediction)),
            }
        )
    return detail


def create_readiness_report(
    metrics_frame: pd.DataFrame,
    sensitivity_frame: pd.DataFrame,
) -> dict[str, Any]:
    test_metrics = metrics_frame.loc[metrics_frame["split"] == "test"]
    comparisons: list[dict[str, Any]] = []
    for (app_id, target), group in test_metrics.groupby(["app_id", "target"]):
        by_model = group.set_index("model_type")
        candidate_mae = float(by_model.loc["candidate_aware_xgboost", "mae"])
        state_mae = float(by_model.loc["state_only_xgboost", "mae"])
        mean_mae = float(by_model.loc["training_mean", "mae"])
        candidate_vs_mean = (mean_mae - candidate_mae) / mean_mae if mean_mae else 0.0
        candidate_vs_state = (state_mae - candidate_mae) / state_mae if state_mae else 0.0
        sensitivity = sensitivity_frame.loc[
            (sensitivity_frame["app_id"] == app_id)
            & (sensitivity_frame["target"] == target),
            "prediction_range",
        ]
        median_range = float(sensitivity.median()) if len(sensitivity) else 0.0
        comparisons.append(
            {
                "app_id": int(app_id),
                "target": target,
                "candidate_test_mae": candidate_mae,
                "state_only_test_mae": state_mae,
                "training_mean_test_mae": mean_mae,
                "relative_improvement_vs_mean": candidate_vs_mean,
                "relative_improvement_vs_state_only": candidate_vs_state,
                "median_fixed_state_prediction_range": median_range,
                "beats_mean": bool(candidate_mae < mean_mae),
                "beats_state_only": bool(candidate_mae < state_mae),
            }
        )

    quality_pass = all(item["beats_mean"] for item in comparisons)
    state_improvements = [item["relative_improvement_vs_state_only"] for item in comparisons]
    candidate_signal_pass = (
        sum(value > 0.01 for value in state_improvements) >= 4
        and float(np.mean(state_improvements)) > 0.01
        and min(state_improvements) > -0.05
    )
    sensitivity_pass = all(
        item["median_fixed_state_prediction_range"] > 1e-6 for item in comparisons
    )

    if quality_pass and candidate_signal_pass and sensitivity_pass:
        status = "READY_FOR_OPTIMIZER_INTEGRATION_TEST"
        interpretation = (
            "Held-out prediction is useful and candidate features add measurable "
            "information beyond workload/state for most app-target models."
        )
    elif quality_pass:
        status = "PREDICTIVE_BUT_CANDIDATE_ADVANTAGE_NOT_ESTABLISHED"
        interpretation = (
            "The models predict held-out performance, but this dataset does not yet "
            "show a consistent advantage from candidate-configuration features."
        )
    else:
        status = "NOT_READY_FOR_OPTIMIZER_INTEGRATION"
        interpretation = (
            "At least one app-target model does not beat the training-mean baseline "
            "on the untouched test iteration."
        )

    return {
        "status": status,
        "interpretation": interpretation,
        "criteria": {
            "all_candidate_models_beat_training_mean": quality_pass,
            "candidate_features_add_consistent_test_value": candidate_signal_pass,
            "predictions_change_when_candidate_changes": sensitivity_pass,
        },
        "comparison_details": comparisons,
        "important_caution": (
            "Candidate sensitivity is a model diagnostic, not proof of causal effect. "
            "Do not tune hyperparameters after inspecting the test iteration."
        ),
    }


def make_prediction_figure(predictions: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)
    for column, app_id in enumerate(sorted(predictions["app_id"].unique())):
        app_frame = predictions.loc[predictions["app_id"] == app_id]
        for row, (target_key, _) in enumerate(TARGETS.items()):
            ax = axes[row, column]
            actual = app_frame[f"actual_{target_key}"]
            predicted = app_frame[f"predicted_{target_key}"]
            low = float(min(actual.min(), predicted.min()))
            high = float(max(actual.max(), predicted.max()))
            ax.scatter(actual, predicted, s=10, alpha=0.45, color="#2463A7", linewidths=0)
            ax.plot([low, high], [low, high], color="#C43B3B", linewidth=1.2)
            score = metrics(actual.to_numpy(), predicted.to_numpy())
            ax.set_title(f"App {app_id} — {target_key.replace('_', ' ')}")
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            ax.text(
                0.04,
                0.96,
                f"MAE={score['mae']:.4f}\nR²={score['r2']:.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            )
            ax.grid(alpha=0.2)
    fig.suptitle("Candidate-aware XGBoost: untouched test iteration", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    frames, candidate_features = load_and_validate(args.data_dir)
    state_only_features = [
        feature for feature in candidate_features if not feature.startswith("candidate_")
    ]
    app_ids = sorted(int(v) for v in frames["train"]["app_id"].unique())

    metric_rows: list[dict[str, Any]] = []
    hyperparameter_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    manifest_models: list[dict[str, Any]] = []

    for app_id in app_ids:
        app_frames = {
            split: frames[split].loc[frames[split]["app_id"] == app_id].copy()
            for split in SPLITS
        }
        prediction_frame = app_frames["test"][
            ["run_id", "num_devices", "window_id", "window_start", "window_end", "app_id"]
        ].copy()
        prediction_frame["load_band"] = load_band(prediction_frame["num_devices"])

        for target_key, target_column in TARGETS.items():
            train_mean = float(app_frames["train"][target_column].mean())
            append_metric_rows(
                metric_rows,
                frames,
                app_id,
                target_key,
                target_column,
                "training_mean",
                train_mean,
                None,
            )

            state_model, state_params, state_val_mae = select_model(
                app_frames["train"],
                app_frames["validation"],
                state_only_features,
                target_column,
                target_key,
            )
            append_metric_rows(
                metric_rows,
                frames,
                app_id,
                target_key,
                target_column,
                "state_only_xgboost",
                state_model,
                state_only_features,
            )
            hyperparameter_rows.append(
                {
                    "app_id": app_id,
                    "target": target_key,
                    "model_type": "state_only_xgboost",
                    "validation_mae": state_val_mae,
                    "best_iteration": int(state_model.best_iteration),
                    **state_params,
                }
            )

            candidate_model, candidate_params, candidate_val_mae = select_model(
                app_frames["train"],
                app_frames["validation"],
                candidate_features,
                target_column,
                target_key,
            )
            append_metric_rows(
                metric_rows,
                frames,
                app_id,
                target_key,
                target_column,
                "candidate_aware_xgboost",
                candidate_model,
                candidate_features,
            )
            hyperparameter_rows.append(
                {
                    "app_id": app_id,
                    "target": target_key,
                    "model_type": "candidate_aware_xgboost",
                    "validation_mae": candidate_val_mae,
                    "best_iteration": int(candidate_model.best_iteration),
                    **candidate_params,
                }
            )

            model_name = f"xgb_app_{app_id}_{target_key}.json"
            candidate_model.get_booster().save_model(str(model_dir / model_name))
            manifest_models.append(
                {
                    "app_id": app_id,
                    "target": target_key,
                    "target_column": target_column,
                    "model_file": f"models/{model_name}",
                    "feature_count": len(candidate_features),
                    "features": candidate_features,
                    "best_iteration": int(candidate_model.best_iteration),
                    "selected_parameters": candidate_params,
                }
            )

            gain = candidate_model.get_booster().get_score(importance_type="gain")
            total_gain = float(sum(gain.values()))
            for feature, value in sorted(gain.items(), key=lambda item: item[1], reverse=True):
                importance_rows.append(
                    {
                        "app_id": app_id,
                        "target": target_key,
                        "feature": feature,
                        "gain": float(value),
                        "normalized_gain": float(value / total_gain) if total_gain else 0.0,
                    }
                )

            sensitivity_rows.extend(
                candidate_sensitivity(
                    candidate_model,
                    candidate_features,
                    app_frames["train"],
                    app_frames["test"],
                    app_id,
                    target_key,
                )
            )

            test_actual = app_frames["test"][target_column].to_numpy()
            test_prediction = clipped_prediction(
                target_key,
                candidate_model.predict(app_frames["test"][candidate_features]),
            )
            state_prediction = clipped_prediction(
                target_key,
                state_model.predict(app_frames["test"][state_only_features]),
            )
            prediction_frame[f"actual_{target_key}"] = test_actual
            prediction_frame[f"predicted_{target_key}"] = test_prediction
            prediction_frame[f"state_only_predicted_{target_key}"] = state_prediction
            prediction_frame[f"training_mean_{target_key}"] = train_mean

        prediction_parts.append(prediction_frame)

    metrics_frame = pd.DataFrame(metric_rows)
    hyperparameters_frame = pd.DataFrame(hyperparameter_rows)
    importance_frame = pd.DataFrame(importance_rows)
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    predictions_frame = pd.concat(prediction_parts, ignore_index=True)

    # Add final test metrics by load band for paper-oriented error inspection.
    band_rows: list[dict[str, Any]] = []
    for (app_id, band), group in predictions_frame.groupby(
        ["app_id", "load_band"], observed=True
    ):
        for target_key in TARGETS:
            row = {
                "app_id": int(app_id),
                "load_band": str(band),
                "target": target_key,
                "rows": int(len(group)),
            }
            row.update(
                metrics(
                    group[f"actual_{target_key}"].to_numpy(),
                    group[f"predicted_{target_key}"].to_numpy(),
                )
            )
            band_rows.append(row)

    readiness = create_readiness_report(metrics_frame, sensitivity_frame)
    candidate_sets = {
        split: set(
            map(
                tuple,
                frames[split][CANDIDATE_COLUMNS]
                .drop_duplicates()
                .round(12)
                .to_numpy(),
            )
        )
        for split in SPLITS
    }
    manifest = {
        "format_version": 1,
        "random_seed": SEED,
        "row_counts": {split: int(len(frames[split])) for split in SPLITS},
        "split_run_ids": {
            split: sorted(frames[split]["run_id"].unique().tolist()) for split in SPLITS
        },
        "candidate_feature_support_in_training": {
            feature: {
                "min": float(frames["train"][feature].min()),
                "max": float(frames["train"][feature].max()),
                "unique_values": int(frames["train"][feature].nunique()),
            }
            for feature in CANDIDATE_COLUMNS
        },
        "candidate_configuration_generalization_check": {
            "unique_train_configurations": len(candidate_sets["train"]),
            "unique_validation_configurations": len(candidate_sets["validation"]),
            "unique_test_configurations": len(candidate_sets["test"]),
            "exact_validation_configurations_seen_in_training": len(
                candidate_sets["validation"] & candidate_sets["train"]
            ),
            "exact_test_configurations_seen_in_training": len(
                candidate_sets["test"] & candidate_sets["train"]
            ),
        },
        "library_versions": {
            "python_note": "See the interpreter used to run this script.",
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "models": manifest_models,
    }

    metrics_frame.to_csv(args.output_dir / "model_metrics.csv", index=False)
    pd.DataFrame(band_rows).to_csv(args.output_dir / "test_metrics_by_load_band.csv", index=False)
    hyperparameters_frame.to_csv(args.output_dir / "selected_hyperparameters.csv", index=False)
    importance_frame.to_csv(args.output_dir / "feature_importance_gain.csv", index=False)
    sensitivity_frame.to_csv(args.output_dir / "candidate_sensitivity.csv", index=False)
    predictions_frame.to_csv(args.output_dir / "test_predictions.csv", index=False)
    (args.output_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.output_dir / "readiness_report.json").write_text(
        json.dumps(readiness, indent=2), encoding="utf-8"
    )
    make_prediction_figure(
        predictions_frame, args.output_dir / "held_out_prediction_diagnostics.png"
    )

    print("Candidate-aware XGBoost training completed.")
    print(f"  Output directory: {args.output_dir.resolve()}")
    print(f"  Models: {len(manifest_models)}")
    print(f"  Readiness status: {readiness['status']}")
    print(f"  Interpretation: {readiness['interpretation']}")


if __name__ == "__main__":
    main()
