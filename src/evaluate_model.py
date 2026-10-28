from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from src.modeling import CATEGORICAL_FEATURES, FEATURE_NAMES, prepare_training_data

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
TARGET_RECALL = 0.75
STRATEGIES = ("unweighted", "class_weighted", "smotenc")


def make_model(strategy: str, positive_count: int, negative_count: int) -> XGBClassifier:
    parameters: dict[str, Any] = {
        "n_estimators": 160,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    if strategy == "class_weighted":
        parameters["scale_pos_weight"] = negative_count / positive_count
    return XGBClassifier(**parameters)


def split_indices(
    groups: pd.Series,
    appointment_dates: pd.Series,
    split_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(groups))
    if split_type == "patient_grouped":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(indices, groups=groups))
        return train_idx, test_idx

    unique_dates = np.sort(appointment_dates.unique())
    if len(unique_dates) < 5:
        raise ValueError("A time-based holdout requires at least five distinct appointment dates.")
    cutoff_index = min(max(int(len(unique_dates) * 0.8), 1), len(unique_dates) - 1)
    cutoff = unique_dates[cutoff_index]
    train_idx = indices[appointment_dates.to_numpy() < cutoff]
    test_idx = indices[appointment_dates.to_numpy() >= cutoff]
    if not len(train_idx) or not len(test_idx):
        raise ValueError("Could not create non-empty chronological train and test splits.")
    return train_idx, test_idx


def split_fit_calibration_and_validation(
    train_idx: np.ndarray,
    groups: pd.Series,
    appointment_dates: pd.Series,
    split_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_type == "patient_grouped":
        outer_splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=RANDOM_STATE + 1
        )
        fit_positions, heldout_positions = next(
            outer_splitter.split(train_idx, groups=groups.iloc[train_idx])
        )
        fit_idx = train_idx[fit_positions]
        heldout_idx = train_idx[heldout_positions]
        inner_splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.5, random_state=RANDOM_STATE + 2
        )
        calibration_positions, validation_positions = next(
            inner_splitter.split(heldout_idx, groups=groups.iloc[heldout_idx])
        )
        return (
            fit_idx,
            heldout_idx[calibration_positions],
            heldout_idx[validation_positions],
        )

    train_dates = appointment_dates.iloc[train_idx].to_numpy()
    unique_dates = np.sort(np.unique(train_dates))
    if len(unique_dates) < 5:
        raise ValueError("A time-based validation split requires at least five training dates.")
    fit_cutoff_index = min(max(int(len(unique_dates) * 0.75), 1), len(unique_dates) - 2)
    validation_cutoff_index = min(
        max(int(len(unique_dates) * 0.875), fit_cutoff_index + 1),
        len(unique_dates) - 1,
    )
    fit_cutoff = unique_dates[fit_cutoff_index]
    validation_cutoff = unique_dates[validation_cutoff_index]
    fit_idx = train_idx[train_dates < fit_cutoff]
    calibration_idx = train_idx[
        (train_dates >= fit_cutoff) & (train_dates < validation_cutoff)
    ]
    validation_idx = train_idx[train_dates >= validation_cutoff]
    if not len(fit_idx) or not len(calibration_idx) or not len(validation_idx):
        raise ValueError("Could not create non-empty chronological training and validation splits.")
    return fit_idx, calibration_idx, validation_idx


def choose_recall_threshold(
    labels: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    target_recall: float = TARGET_RECALL,
) -> float:
    _, recall, thresholds = precision_recall_curve(labels, probabilities)
    eligible = thresholds[recall[:-1] >= target_recall]
    return float(eligible.max()) if len(eligible) else 0.0


def expected_calibration_error(
    labels: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    labels_array = np.asarray(labels)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        in_bin = (probabilities >= edges[index]) & (
            probabilities < edges[index + 1] if index < bins - 1 else probabilities <= 1.0
        )
        if in_bin.any():
            ece += in_bin.mean() * abs(labels_array[in_bin].mean() - probabilities[in_bin].mean())
    return float(ece)


def score_predictions(
    labels: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predicted = probabilities >= threshold
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "expected_calibration_error": expected_calibration_error(labels, probabilities),
        "threshold": float(threshold),
        "precision_at_threshold": float(precision_score(labels, predicted, zero_division=0)),
        "recall_at_threshold": float(recall_score(labels, predicted, zero_division=0)),
        "f1_at_threshold": float(f1_score(labels, predicted, zero_division=0)),
        "accuracy_at_threshold": float(accuracy_score(labels, predicted)),
    }


def fit_strategy(
    strategy: str,
    features: pd.DataFrame,
    labels: pd.Series,
    fit_idx: np.ndarray,
    calibration_idx: np.ndarray,
    validation_idx: np.ndarray,
    test_idx: np.ndarray,
    target_recall: float,
) -> tuple[Any, dict[str, float], dict[str, float]]:
    y_fit = labels.iloc[fit_idx]
    model = make_model(
        strategy,
        positive_count=int(y_fit.sum()),
        negative_count=int((y_fit == 0).sum()),
    )
    x_fit = features.iloc[fit_idx]
    if strategy == "smotenc":
        sampler = SMOTENC(
            categorical_features=[FEATURE_NAMES.index(name) for name in CATEGORICAL_FEATURES],
            random_state=RANDOM_STATE,
        )
        x_fit, y_fit = sampler.fit_resample(x_fit, y_fit)
        x_fit = pd.DataFrame(x_fit, columns=FEATURE_NAMES)
    model.fit(x_fit, y_fit)

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method="sigmoid",
    )
    calibrated_model.fit(features.iloc[calibration_idx], labels.iloc[calibration_idx])
    validation_probabilities = calibrated_model.predict_proba(features.iloc[validation_idx])[:, 1]
    threshold = choose_recall_threshold(
        labels.iloc[validation_idx], validation_probabilities, target_recall
    )
    test_probabilities = calibrated_model.predict_proba(features.iloc[test_idx])[:, 1]
    return (
        calibrated_model,
        score_predictions(labels.iloc[validation_idx], validation_probabilities, threshold),
        score_predictions(labels.iloc[test_idx], test_probabilities, threshold),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(
    data_path: Path,
    output_dir: Path,
    target_recall: float = TARGET_RECALL,
) -> pd.DataFrame:
    if not 0 < target_recall < 1:
        raise ValueError("The example target recall must be strictly between 0 and 1.")
    raw = pd.read_csv(data_path)
    features, labels, groups, dates, dropped = prepare_training_data(raw)
    results: list[dict[str, Any]] = []
    selected_group_model: Any | None = None
    selected_group_row: dict[str, Any] | None = None

    for split_type in ("patient_grouped", "time_based"):
        train_idx, test_idx = split_indices(groups, dates, split_type)
        fit_idx, calibration_idx, validation_idx = split_fit_calibration_and_validation(
            train_idx, groups, dates, split_type
        )
        validation_candidates: list[tuple[float, Any, dict[str, Any]]] = []
        for strategy in STRATEGIES:
            model, validation_metrics, test_metrics = fit_strategy(
                strategy,
                features,
                labels,
                fit_idx,
                calibration_idx,
                validation_idx,
                test_idx,
                target_recall,
            )
            row = {
                "split": split_type,
                "strategy": strategy,
                "fit_rows": int(len(fit_idx)),
                "calibration_rows": int(len(calibration_idx)),
                "validation_rows": int(len(validation_idx)),
                "test_rows": int(len(test_idx)),
                "validation_average_precision": validation_metrics["average_precision"],
                "validation_brier_score": validation_metrics["brier_score"],
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
            results.append(row)
            print(
                f"{split_type} / {strategy}: "
                f"test ROC-AUC={test_metrics['roc_auc']:.3f}, "
                f"PR-AUC={test_metrics['average_precision']:.3f}, "
                f"Brier={test_metrics['brier_score']:.3f}, "
                f"recall={test_metrics['recall_at_threshold']:.3f} "
                f"at validation-derived threshold={validation_metrics['threshold']:.3f}"
            )
            if split_type == "patient_grouped":
                validation_candidates.append(
                    (validation_metrics["average_precision"], model, row)
                )
        if split_type == "patient_grouped":
            _, selected_group_model, selected_group_row = max(
                validation_candidates, key=lambda candidate: candidate[0]
            )

    if selected_group_model is None or selected_group_row is None:
        raise RuntimeError("No patient-grouped model was available to save.")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgboost_model.pkl"
    feature_path = output_dir / "feature_names.pkl"
    joblib.dump(selected_group_model, model_path)
    joblib.dump(FEATURE_NAMES, feature_path)

    base_model = selected_group_model.estimator.estimator
    explainer_path = output_dir / "shap_explainer.pkl"
    import shap

    shap_explainer = shap.TreeExplainer(base_model)
    joblib.dump(shap_explainer, explainer_path)

    evaluation_frame = pd.DataFrame(results)
    report_path = output_dir.parent / "reports" / "model_evaluation.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_frame.to_csv(report_path, index=False)
    metadata = {
        "model_sha256": file_sha256(model_path),
        "explainer_model_sha256": file_sha256(model_path),
        "feature_names": FEATURE_NAMES,
        "selected_strategy": selected_group_row["strategy"],
        "selected_on": "highest validation average precision in patient-grouped split",
        "threshold": selected_group_row["validation_threshold"],
        "target_recall": target_recall,
        "threshold_source": "patient-grouped validation split; illustrative screening target only",
        "patient_grouped_test": {
            "roc_auc": selected_group_row["test_roc_auc"],
            "average_precision": selected_group_row["test_average_precision"],
            "brier_score": selected_group_row["test_brier_score"],
            "recall_at_threshold": selected_group_row["test_recall_at_threshold"],
        },
        "data_rows": len(features),
        "dropped_rows": dropped,
        "calibration": "sigmoid calibration fitted on validation data",
        "not_for_clinical_use": True,
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSelected strategy: {metadata['selected_strategy']}")
    print(f"Saved artifacts to: {output_dir}")
    print(f"Evaluation report: {report_path}")
    return evaluation_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare calibrated XGBoost imbalance strategies with grouped and time holdouts."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "KaggleV2-May-2016.csv",
        help="Path to the original Kaggle CSV.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--target-recall",
        type=float,
        default=TARGET_RECALL,
        help="Illustrative recall target used to select a threshold on validation data.",
    )
    arguments = parser.parse_args()
    evaluate(arguments.data, arguments.output, arguments.target_recall)


if __name__ == "__main__":
    main()
