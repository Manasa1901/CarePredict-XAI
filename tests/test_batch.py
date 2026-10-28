import unittest

import numpy as np
import pandas as pd

from src.batch import (
    batch_summary,
    build_prediction_report,
    classify_risk,
    drift_report,
    population_stability_index,
    validate_batch_frame,
)
from src.modeling import FEATURE_NAMES
from src.persistence import connect
from unittest.mock import patch


def valid_batch(rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [30] * rows,
            "Scholarship": [0] * rows,
            "Hypertension": [0] * rows,
            "Diabetes": [0] * rows,
            "Alcoholism": [0] * rows,
            "Handicap": [0] * rows,
            "SMS_received": [1] * rows,
            "Days_In_Advance": [7] * rows,
            "Appointment_Weekday": [2] * rows,
            "Appointment_Month": [4] * rows,
            "Scheduled_Hour": [10] * rows,
            "Gender_M": [0] * rows,
        },
        columns=FEATURE_NAMES,
    )


class BatchPredictionTests(unittest.TestCase):
    def test_batch_schema_and_ranges_are_checked(self):
        validated = validate_batch_frame(valid_batch())
        self.assertEqual(validated.columns.tolist(), FEATURE_NAMES)
        self.assertEqual(len(validated), 3)

        with_identifiers = valid_batch()
        with_identifiers["PatientId"] = ["123", "456", "789"]
        with self.assertRaisesRegex(ValueError, "identifiers"):
            validate_batch_frame(with_identifiers)

        invalid = valid_batch()
        invalid.loc[0, "Appointment_Weekday"] = 7
        with self.assertRaisesRegex(ValueError, "between 0 and 6"):
            validate_batch_frame(invalid)

    def test_risk_bands_match_requested_thresholds(self):
        risk = classify_risk([0.0999, 0.1, 0.1885, 0.1886], 0.1885)
        self.assertEqual(risk, ["Low", "Medium", "High", "High"])

    def test_report_contains_no_input_features_or_identifiers(self):
        report = build_prediction_report(
            [0.05, 0.15, 0.4],
            high_threshold=0.1885,
            model_version="abcdef123456",
        )
        self.assertEqual(report["Risk Band"].tolist(), ["Low", "Medium", "High"])
        self.assertEqual(report["Model Version"].iloc[0], "abcdef123456")
        self.assertTrue(set(FEATURE_NAMES).isdisjoint(report.columns))

    def test_summary_is_aggregate_only(self):
        report = build_prediction_report([0.05, 0.15, 0.4], 0.1885, "abc")
        summary = batch_summary(report, "batch", "abc")
        self.assertEqual(
            (summary["Total"], summary["Low"], summary["Medium"], summary["High"]),
            (3, 1, 1, 1),
        )
        self.assertEqual(set(summary), {
            "Prediction Date (UTC)", "Model Version", "Source",
            "Total", "Low", "Medium", "High",
        })

    def test_drift_report_handles_constant_columns_and_detects_shift(self):
        reference = valid_batch(50)
        current = valid_batch(50)
        current["Age"] = 75
        report = drift_report(reference, current)
        age_psi = report.loc[report["Feature"] == "Age", "Population Stability Index"].iloc[0]
        self.assertTrue(np.isfinite(age_psi))
        self.assertGreater(age_psi, 0.25)
        self.assertTrue(np.isfinite(population_stability_index(
            pd.Series([1, 1, 1]), pd.Series([1, 1, 1])
        )))

    def test_remote_postgres_requires_verified_tls(self):
        with patch.dict(
            "os.environ",
            {
                "CARE_PREDICT_DATABASE_URL":
                    "host=db.example.test dbname=carepredict user=app sslmode=require"
            },
        ):
            with self.assertRaisesRegex(ValueError, "sslmode=verify-full"):
                connect()

    def test_database_url_is_not_required_when_history_is_unconfigured(self):
        with patch.dict("os.environ", {}, clear=False):
            with patch("src.persistence.os.environ.get", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "CARE_PREDICT_DATABASE_URL"):
                    connect()


if __name__ == "__main__":
    unittest.main()
