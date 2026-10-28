# CarePredict-XAI

A Streamlit portfolio prototype that explores appointment no-show prediction and model explanations using the public Kaggle Medical Appointment No Shows dataset. It is not a clinical product or a validated care decision tool.

## Dashboard, batch predictions, and reports

- The hospital dashboard shows sample-dataset appointment and attendance trends, plus the total number scored and risk-band distribution for this session or the optional PostgreSQL aggregate audit history.
- The single-prediction page estimates no-show probability. The batch page accepts feature-only CSV files with exactly the 12 model columns and returns low (<10%), medium (>=10% and below the model's illustrative threshold), or high (>= threshold) risk bands. The default example cutoff is about 18.9%.
- Batch uploads are limited to 5,000 rows and 5 MB. Identifiers, outcome labels, and extra columns are rejected. The app asks the operator to attest the data is synthetic/de-identified. The bundled CSV template is `reports\batch_input_template.csv`.
- Prediction CSV reports include row position, timestamp, model version, probability, risk band, and the top three SHAP feature contributions when the explainer is available. The uploaded feature values and direct identifiers are not included in report exports.
- Per-row report/history data stays in the current Streamlit session. Optional PostgreSQL stores only run timestamp, model version, single/batch source, and aggregate low/medium/high counts; it never stores input rows or identifiers. Aggregate records are retained for 90 days. No app-user identity is recorded.
- The Data & Validation page can calculate heuristic population-stability-index (PSI) input-distribution comparisons against the bundled public sample and evaluate ROC-AUC, average precision, and Brier score from a separate labeled dataset. Use only synthetic or properly de-identified, authorized data. Uploaded features/outcomes and these evaluation results are not persisted.

PSI is an input-shift alert, not an estimate of model accuracy or performance drift. External validation needs a genuinely independent, appropriately authorized dataset; none has been supplied with this project. A prospective impact study also needs to compare an approved follow-up strategy with usual care and collect outcomes under a predefined protocol. This app does not run interventions or establish that outreach causes attendance improvement.

## Optional PostgreSQL aggregate history

PostgreSQL is optional. Create a dedicated database and app role, apply `db\schema.sql` as an administrator, then grant the app role only the listed table/sequence permissions. Configure its connection string in the process environment; do not commit credentials or put them in the uploaded CSV:

```powershell
$env:CARE_PREDICT_DATABASE_URL = "postgresql://carepredict_app:YOUR_PASSWORD@localhost:5432/carepredict?sslmode=disable"
```

For a remote PostgreSQL host, the app requires `sslmode=verify-full` and a trusted CA certificate. Configure the DSN in an approved secrets manager/environment; do not paste it into source files or chat. A missing/unavailable PostgreSQL connection is reported by the app; without a DSN, history is session-only.

## Setup

Use Python 3.10 or later. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Keep the original `data\KaggleV2-May-2016.csv` in place. The training pipeline reads this source CSV and derives all model features itself.

## Evaluate and train

Run the evaluation pipeline from the project root:

```powershell
python -m src.evaluate_model
```

The pipeline compares unweighted training, class weighting, and `SMOTENC` on:

- A patient-grouped holdout, keeping each patient's appointments together.
- A chronological holdout, keeping later appointment dates out of training.

For each split, the fit, probability-calibration, threshold-selection, and test partitions are kept separate. The sigmoid probability calibrator is fitted on calibration data. The example screening threshold is chosen on a separate validation partition to target 75% recall by default; change this explicitly with `--target-recall 0.80` if you want to explore another target. The strategy is chosen by validation average precision, never by test performance.

Test reports include ROC-AUC, average precision (PR-AUC), Brier score, log loss, expected calibration error, and precision/recall/F1 at the validation-derived threshold. Metrics and thresholds are estimates for this dataset/split, not performance guarantees. The default 75% recall target is illustrative and has not been selected with care-team input.

The run updates the model artifacts under `models\` and writes `reports\model_evaluation.csv`. It also creates a SHAP explainer for the selected underlying XGBoost model. Review the report before using the demo; the app displays the patient-grouped holdout ROC-AUC and selected example threshold.

## Run the dashboard

After model artifacts have been generated:

```powershell
python -m streamlit run app\app.py
```

For the selected local-only prototype scope, bind Streamlit to loopback:

```powershell
python -m streamlit run app\app.py --server.address 127.0.0.1
```

The dashboard checks model, feature, and metadata compatibility at startup. It refuses to use a SHAP explainer whose recorded model fingerprint does not match the current model. SHAP describes the underlying tree score before probability calibration; what-if comparisons show model associations, not causal effects.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Safety and privacy limits

- This implementation is restricted to local demonstration with synthetic or properly de-identified data. Do not upload real patient records or identifiers. The checkbox is an operator attestation, not a technical de-identification guarantee.
- The model has not undergone clinical validation, prospective evaluation, external validation, fairness review, or deployment monitoring.
- No authentication or authorization is implemented. Keep the server bound to localhost and do not expose it on a network. The optional PostgreSQL metadata log is not a clinical audit system and contains no operator identity.
- The PostgreSQL option persists only aggregate run counts and model-version metadata for up to 90 days; per-row prediction reports remain temporary in session memory and can be downloaded by the local operator.
- Predictions and screening flags must not be used to diagnose, deny, prioritize, or automate care. Any real-world research or deployment requires appropriate privacy/security review, governance, representative external evaluation, bias analysis, calibration monitoring, and qualified clinical/operational oversight.
- Model artifacts use Python pickle/joblib serialization. Only load artifacts you generated or otherwise trust; loading an untrusted pickle can execute code.

## Real-world deployment boundary

This project remains a local-only prototype, not an approved system for real patient data. The optional PostgreSQL feature is limited to aggregate metadata and does not add authentication. Before any real-data or multi-user deployment, obtain organizational privacy/security approval and implement an established identity provider, role-based authorization, secure session management, HTTPS, encryption, access/audit controls, secrets management, and a reviewed retention/deletion policy. Authentication alone does not make health-data processing safe or compliant. Newer facility/time-period validation and any prospective impact evaluation require appropriately authorized data and qualified oversight.

## Project layout

- `app\app.py` — Streamlit dashboard
- `src\modeling.py` — shared feature engineering and model schema
- `src\evaluate_model.py` — grouped/time-based evaluation and artifact training
- `notebooks\01_data_understanding.ipynb` — exploratory notebook
- `tests\` — focused pipeline unit tests
