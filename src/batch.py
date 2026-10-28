from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.modeling import CATEGORICAL_FEATURES, FEATURE_NAMES

LOW_RISK_CUTOFF = 0.10
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_BATCH_ROWS = 5000
FORBIDDEN_COLUMN_NAMES = {
    "patientid",
    "patient_id",
    "mrn",
    "medical_record_number",
    "appointmentid",
    "appointment_id",
    "name",
    "first_name",
    "last_name",
    "dob",
    "date_of_birth",
    "birthdate",
    "email",
    "phone",
    "address",
    "ssn",
}
INTEGER_FEATURES = {
    "Age": (0, 120),
    "Scholarship": (0, 1),
    "Hypertension": (0, 1),
    "Diabetes": (0, 1),
    "Alcoholism": (0, 1),
    "Handicap": (0, 4),
    "SMS_received": (0, 1),
    "Days_In_Advance": (0, 365),
    "Appointment_Weekday": (0, 6),
    "Appointment_Month": (1, 12),
    "Scheduled_Hour": (0, 23),
    "Gender_M": (0, 1),
}


def validate_batch_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [str(column).strip() for column in frame.columns]
    normalized_columns = [column.casefold() for column in columns]
    forbidden = sorted(
        column
        for column, normalized in zip(columns, normalized_columns)
        if normalized in FORBIDDEN_COLUMN_NAMES
    )
    if forbidden:
        raise ValueError(
            "The upload contains patient identifiers or direct identifiers. "
            "Remove identifying columns and use only synthetic or de-identified profiles."
        )
    duplicates = sorted(
        {column for column in columns if columns.count(column) > 1}
    )
    if duplicates:
        raise ValueError("The upload has duplicate column names.")
    if set(columns) != set(FEATURE_NAMES):
        missing = sorted(set(FEATURE_NAMES) - set(columns))
        extra = sorted(set(columns) - set(FEATURE_NAMES))
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected: {', '.join(extra)}")
        raise ValueError(
            "Upload columns must exactly match the model feature schema (no IDs or outcomes); "
            + "; ".join(details)
        )
    if not 1 <= len(frame) <= MAX_BATCH_ROWS:
        raise ValueError(f"Upload between 1 and {MAX_BATCH_ROWS:,} rows per batch.")

    clean = frame.copy()
    clean.columns = columns
    for column, (minimum, maximum) in INTEGER_FEATURES.items():
        converted = pd.to_numeric(clean[column], errors="coerce")
        values = converted.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite numeric values in every row.")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain whole-number values.")
        if ((values < minimum) | (values > maximum)).any():
            raise ValueError(f"{column} values must be between {minimum} and {maximum}.")
        clean[column] = converted.astype("int64")
    return clean[FEATURE_NAMES]


def classify_risk(probabilities: Any, high_threshold: float) -> list[str]:
    if not 0.0 < LOW_RISK_CUTOFF < high_threshold < 1.0:
        raise ValueError("Risk thresholds must satisfy 0 < 10% < high threshold < 100%.")
    return [
        "Low" if probability < LOW_RISK_CUTOFF else
        "Medium" if probability < high_threshold else
        "High"
        for probability in np.asarray(probabilities, dtype=float)
    ]


def build_prediction_report(
    probabilities: Any,
    high_threshold: float,
    model_version: str,
    shap_explanation: Any | None = None,
) -> pd.DataFrame:
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise ValueError("Model returned invalid probabilities.")
    risk_bands = classify_risk(probabilities, high_threshold)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = pd.DataFrame(
        {
            "Row": np.arange(1, len(probabilities) + 1),
            "Predicted At (UTC)": generated_at,
            "Model Version": model_version,
            "Estimated No-Show Probability": probabilities,
            "Risk Band": risk_bands,
        }
    )
    report["Estimated No-Show Probability"] = report[
        "Estimated No-Show Probability"
    ].map(lambda probability: round(float(probability), 6))

    if shap_explanation is not None:
        values = np.asarray(shap_explanation.values)
        if values.ndim == 3:
            values = values[:, :, -1]
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape != (len(probabilities), len(FEATURE_NAMES)):
            raise ValueError(
                f"SHAP returned unexpected shape {values.shape} for "
                f"{len(probabilities)} rows and {len(FEATURE_NAMES)} features."
            )
        for rank in range(3):
            feature_names: list[str] = []
            contributions: list[float] = []
            for row in values:
                order = np.argsort(np.abs(row))[::-1]
                feature_index = order[rank]
                feature_names.append(FEATURE_NAMES[feature_index])
                contributions.append(round(float(row[feature_index]), 6))
            report[f"Top factor {rank + 1}"] = feature_names
            report[f"Top factor {rank + 1} SHAP value"] = contributions
    return report


def batch_summary(
    report: pd.DataFrame,
    source: str,
    model_version: str,
) -> dict[str, Any]:
    counts = report["Risk Band"].value_counts()
    return {
        "Prediction Date (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Model Version": model_version,
        "Source": source,
        "Total": int(len(report)),
        "Low": int(counts.get("Low", 0)),
        "Medium": int(counts.get("Medium", 0)),
        "High": int(counts.get("High", 0)),
    }


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    categorical: bool = False,
) -> float:
    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(reference_values) or not len(current_values):
        return float("nan")
    if categorical:
        categories = np.union1d(np.unique(reference_values), np.unique(current_values))
        reference_frequency = np.array(
            [(reference_values == category).mean() for category in categories]
        )
        current_frequency = np.array(
            [(current_values == category).mean() for category in categories]
        )
    else:
        quantiles = np.unique(np.quantile(reference_values, np.linspace(0, 1, 11)))
        if len(quantiles) < 3:
            categories = np.union1d(np.unique(reference_values), np.unique(current_values))
            reference_frequency = np.array(
                [(reference_values == category).mean() for category in categories]
            )
            current_frequency = np.array(
                [(current_values == category).mean() for category in categories]
            )
        else:
            quantiles[0] = -np.inf
            quantiles[-1] = np.inf
            reference_frequency = (
                np.histogram(reference_values, bins=quantiles)[0] / len(reference_values)
            )
            current_frequency = (
                np.histogram(current_values, bins=quantiles)[0] / len(current_values)
            )
    epsilon = 1e-6
    reference_frequency = np.clip(reference_frequency, epsilon, None)
    current_frequency = np.clip(current_frequency, epsilon, None)
    return float(
        np.sum(
            (current_frequency - reference_frequency)
            * np.log(current_frequency / reference_frequency)
        )
    )


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for feature in FEATURE_NAMES:
        psi = population_stability_index(
            reference[feature],
            current[feature],
            categorical=feature in CATEGORICAL_FEATURES,
        )
        rows.append(
            {
                "Feature": feature,
                "Population Stability Index": psi,
                "Interpretation": (
                    "High distribution shift"
                    if psi >= 0.25
                    else "Moderate distribution shift"
                    if psi >= 0.10
                    else "Limited distribution shift"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "Population Stability Index", ascending=False
    )
