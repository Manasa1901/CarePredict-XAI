from __future__ import annotations

from typing import Any

import pandas as pd

FEATURE_NAMES = [
    "Age",
    "Scholarship",
    "Hypertension",
    "Diabetes",
    "Alcoholism",
    "Handicap",
    "SMS_received",
    "Days_In_Advance",
    "Appointment_Weekday",
    "Appointment_Month",
    "Scheduled_Hour",
    "Gender_M",
]

CATEGORICAL_FEATURES = [
    "Scholarship",
    "Hypertension",
    "Diabetes",
    "Alcoholism",
    "Handicap",
    "SMS_received",
    "Appointment_Weekday",
    "Appointment_Month",
    "Scheduled_Hour",
    "Gender_M",
]

RAW_COLUMNS = {
    "patient_id": "PatientId",
    "appointment_id": "AppointmentID",
    "scheduled": "ScheduledDay",
    "appointment": "AppointmentDay",
    "age": "Age",
    "scholarship": "Scholarship",
    "hypertension": "Hipertension",
    "diabetes": "Diabetes",
    "alcoholism": "Alcoholism",
    "handicap": "Handcap",
    "sms": "SMS_received",
    "gender": "Gender",
    "target": "No-show",
}


def prepare_training_data(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, dict[str, int]]:
    """Create serving-compatible features and discard invalid source rows."""
    missing = sorted(set(RAW_COLUMNS.values()) - set(raw.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    frame = raw.copy()
    scheduled = pd.to_datetime(frame[RAW_COLUMNS["scheduled"]], errors="coerce", utc=True)
    appointment = pd.to_datetime(frame[RAW_COLUMNS["appointment"]], errors="coerce", utc=True)
    days_in_advance = (
        appointment.dt.normalize() - scheduled.dt.normalize()
    ).dt.days
    valid_target = frame[RAW_COLUMNS["target"]].isin(["No", "Yes"])
    valid_dates = scheduled.notna() & appointment.notna()
    valid_age = frame[RAW_COLUMNS["age"]].between(0, 120)
    valid_wait = days_in_advance.ge(0)
    valid_patient = frame[RAW_COLUMNS["patient_id"]].notna()
    valid_appointment = frame[RAW_COLUMNS["appointment_id"]].notna()
    valid_gender = frame[RAW_COLUMNS["gender"]].isin(["F", "M"])
    duplicate_appointment = frame[RAW_COLUMNS["appointment_id"]].duplicated(keep="first")
    valid = (
        valid_target
        & valid_dates
        & valid_age
        & valid_wait
        & valid_patient
        & valid_appointment
        & valid_gender
        & ~duplicate_appointment
    )
    dropped = {
        "invalid_target": int((~valid_target).sum()),
        "invalid_dates": int((~valid_dates).sum()),
        "invalid_age": int((valid_target & valid_dates & ~valid_age).sum()),
        "negative_wait": int((valid_target & valid_dates & valid_age & ~valid_wait).sum()),
        "missing_patient_id": int((~valid_patient).sum()),
        "missing_appointment_id": int((~valid_appointment).sum()),
        "invalid_gender": int((~valid_gender).sum()),
        "duplicate_appointments": int(duplicate_appointment.sum()),
        "rows_discarded": int((~valid).sum()),
    }

    frame = frame.loc[valid].copy()
    scheduled = scheduled.loc[valid]
    appointment = appointment.loc[valid]
    days_in_advance = days_in_advance.loc[valid]
    features = pd.DataFrame(
        {
            "Age": frame[RAW_COLUMNS["age"]].astype(int),
            "Scholarship": frame[RAW_COLUMNS["scholarship"]].astype(int),
            "Hypertension": frame[RAW_COLUMNS["hypertension"]].astype(int),
            "Diabetes": frame[RAW_COLUMNS["diabetes"]].astype(int),
            "Alcoholism": frame[RAW_COLUMNS["alcoholism"]].astype(int),
            "Handicap": frame[RAW_COLUMNS["handicap"]].astype(int),
            "SMS_received": frame[RAW_COLUMNS["sms"]].astype(int),
            "Days_In_Advance": days_in_advance.astype(int),
            "Appointment_Weekday": appointment.dt.dayofweek.astype(int),
            "Appointment_Month": appointment.dt.month.astype(int),
            "Scheduled_Hour": scheduled.dt.hour.astype(int),
            "Gender_M": frame[RAW_COLUMNS["gender"]].eq("M").astype(int),
        },
        index=frame.index,
    )
    labels = frame[RAW_COLUMNS["target"]].map({"No": 0, "Yes": 1}).astype(int)
    groups = frame[RAW_COLUMNS["patient_id"]].astype(str)
    appointment_dates = appointment.dt.normalize()
    return (
        features[FEATURE_NAMES].reset_index(drop=True),
        labels.reset_index(drop=True),
        groups.reset_index(drop=True),
        appointment_dates.reset_index(drop=True),
        dropped,
    )


def validate_feature_schema(
    feature_names: list[str], model: Any
) -> None:
    if feature_names != FEATURE_NAMES:
        raise ValueError(
            "Saved feature names do not match the application schema. "
            f"Expected {FEATURE_NAMES}; found {feature_names}."
        )
    model_feature_count = getattr(model, "n_features_in_", None)
    if model_feature_count is not None and model_feature_count != len(FEATURE_NAMES):
        raise ValueError(
            f"Model expects {model_feature_count} features; "
            f"the application provides {len(FEATURE_NAMES)}."
        )
