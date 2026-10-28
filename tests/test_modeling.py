import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.evaluate_model import (
    choose_recall_threshold,
    expected_calibration_error,
    split_fit_calibration_and_validation,
    split_indices,
)
from src.modeling import FEATURE_NAMES, prepare_training_data, validate_feature_schema


def sample_raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PatientId": [1, 2, 3, 4, 5],
            "AppointmentID": [11, 12, 13, 14, 15],
            "Gender": ["F", "M", "F", "M", "F"],
            "ScheduledDay": [
                "2016-04-29 18:30:00+00:00",
                "2016-04-28 10:00:00+00:00",
                "2016-04-29 10:00:00+00:00",
                "2016-04-28 10:00:00+00:00",
                "2016-04-27 20:00:00+00:00",
            ],
            "AppointmentDay": [
                "2016-04-29 00:00:00+00:00",
                "2016-04-29 00:00:00+00:00",
                "2016-04-28 00:00:00+00:00",
                "2016-04-29 00:00:00+00:00",
                "2016-04-29 00:00:00+00:00",
            ],
            "Age": [35, 40, 20, -1, 52],
            "Neighbourhood": ["A", "B", "A", "C", "D"],
            "Scholarship": [0, 1, 0, 0, 1],
            "Hipertension": [0, 1, 0, 0, 1],
            "Diabetes": [0, 0, 0, 0, 1],
            "Alcoholism": [0, 0, 0, 0, 0],
            "Handcap": [0, 0, 0, 0, 1],
            "SMS_received": [0, 1, 0, 0, 1],
            "No-show": ["No", "Yes", "No", "No", "Yes"],
        }
    )


class PrepareTrainingDataTests(unittest.TestCase):
    def test_calendar_days_avoid_negative_same_day_waits(self):
        features, labels, groups, dates, dropped = prepare_training_data(sample_raw_data())

        self.assertEqual(features["Days_In_Advance"].tolist(), [0, 1, 2])
        self.assertEqual(labels.tolist(), [0, 1, 1])
        self.assertEqual(groups.tolist(), ["1", "2", "5"])
        self.assertEqual(dates.dt.day.tolist(), [29, 29, 29])
        self.assertEqual(dropped["negative_wait"], 1)
        self.assertEqual(dropped["invalid_age"], 1)

    def test_feature_schema_has_stable_order(self):
        features, *_ = prepare_training_data(sample_raw_data())
        self.assertEqual(features.columns.tolist(), FEATURE_NAMES)
        self.assertEqual(features.loc[0, "Gender_M"], 0)
        self.assertEqual(features.loc[1, "Gender_M"], 1)

    def test_missing_source_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            prepare_training_data(pd.DataFrame({"Age": [30]}))


class EvaluationHelpersTests(unittest.TestCase):
    def test_patient_grouped_splits_do_not_share_patients(self):
        groups = pd.Series([f"patient-{index}" for index in range(100)])
        dates = pd.Series(pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC"))
        train_idx, test_idx = split_indices(groups, dates, "patient_grouped")
        fit_idx, calibration_idx, validation_idx = split_fit_calibration_and_validation(
            train_idx, groups, dates, "patient_grouped"
        )

        self.assertFalse(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))
        self.assertFalse(set(groups.iloc[fit_idx]) & set(groups.iloc[calibration_idx]))
        self.assertFalse(set(groups.iloc[fit_idx]) & set(groups.iloc[validation_idx]))
        self.assertFalse(set(groups.iloc[calibration_idx]) & set(groups.iloc[validation_idx]))

    def test_time_splits_keep_later_dates_out_of_training(self):
        groups = pd.Series([f"patient-{index}" for index in range(100)])
        dates = pd.Series(pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC"))
        train_idx, test_idx = split_indices(groups, dates, "time_based")
        fit_idx, calibration_idx, validation_idx = split_fit_calibration_and_validation(
            train_idx, groups, dates, "time_based"
        )

        self.assertLess(dates.iloc[fit_idx].max(), dates.iloc[calibration_idx].min())
        self.assertLess(dates.iloc[calibration_idx].max(), dates.iloc[validation_idx].min())
        self.assertLess(dates.iloc[validation_idx].max(), dates.iloc[test_idx].min())

    def test_recall_threshold_and_calibration_error(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.3, 0.6, 0.9])

        threshold = choose_recall_threshold(labels, probabilities, target_recall=0.5)

        self.assertEqual(threshold, 0.9)
        self.assertAlmostEqual(expected_calibration_error(labels, probabilities), 0.225)

    def test_model_feature_count_is_validated(self):
        model = DummyClassifier().fit(np.zeros((2, len(FEATURE_NAMES))), [0, 1])
        validate_feature_schema(FEATURE_NAMES, model)
        with self.assertRaisesRegex(ValueError, "feature names do not match"):
            validate_feature_schema(FEATURE_NAMES[:-1], model)


if __name__ == "__main__":
    unittest.main()
