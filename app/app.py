from datetime import datetime
import hashlib
import json
import logging
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
import streamlit as st
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from src.batch import (
    MAX_BATCH_ROWS,
    MAX_UPLOAD_BYTES,
    batch_summary,
    build_prediction_report,
    classify_risk,
    drift_report,
    validate_batch_frame,
)
from src.modeling import FEATURE_NAMES, validate_feature_schema
from src.persistence import get_connection_url, recent_events, record_event

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="CarePredict-XAI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "cleaned_appointments.csv"
REFERENCE_DATA_PATH = BASE_DIR / "data" / "KaggleV2-May-2016.csv"
MODEL_VERSION = hashlib.sha256((MODEL_DIR / "xgboost_model.pkl").read_bytes()).hexdigest()[:12]

MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
FEATURE_PATH = MODEL_DIR / "feature_names.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
for required_path in (MODEL_PATH, FEATURE_PATH, METADATA_PATH):
    if not required_path.is_file():
        raise FileNotFoundError(
            f"Required model artifact is missing: {required_path}. "
            "Run `python -m src.evaluate_model` from the project root."
        )

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
model_digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
if metadata.get("model_sha256") != model_digest:
    raise ValueError("The model metadata does not match the saved model artifact. Retrain the model.")
if metadata.get("feature_names") != feature_names:
    raise ValueError("The model metadata and feature-name artifact do not match.")
validate_feature_schema(feature_names, model)
HIGH_RISK_THRESHOLD = float(metadata["threshold"])
TARGET_RECALL = float(metadata["target_recall"])
if not 0.0 <= HIGH_RISK_THRESHOLD <= 1.0:
    raise ValueError(f"Invalid screening threshold in {METADATA_PATH}.")
if not 0.0 < TARGET_RECALL < 1.0:
    raise ValueError(f"Invalid target recall in {METADATA_PATH}.")

try:
    if (
        metadata.get("explainer_model_sha256") == model_digest
        and (MODEL_DIR / "shap_explainer.pkl").is_file()
    ):
        shap_explainer = joblib.load(MODEL_DIR / "shap_explainer.pkl")
        shap_error = None
    else:
        shap_explainer = None
        shap_error = "No SHAP explainer matching the current model was found."
except (
    OSError,
    ImportError,
    ValueError,
    pickle.UnpicklingError,
    EOFError,
    AttributeError,
) as error:
    shap_explainer = None
    shap_error = str(error)

FEATURE_LABELS = {
    "Age": "Age",
    "Scholarship": "Scholarship",
    "Hypertension": "Hypertension",
    "Diabetes": "Diabetes",
    "Alcoholism": "Alcoholism",
    "Handicap": "Handicap",
    "SMS_received": "SMS Reminder",
    "Days_In_Advance": "Days in Advance",
    "Appointment_Weekday": "Appointment Weekday",
    "Appointment_Month": "Appointment Month",
    "Scheduled_Hour": "Scheduled Hour",
    "Gender_M": "Gender: Male",
}
DEFAULT_INPUT = {
    "Age": 30,
    "Scholarship": 0,
    "Hypertension": 0,
    "Diabetes": 0,
    "Alcoholism": 0,
    "Handicap": 0,
    "SMS_received": 1,
    "Days_In_Advance": 7,
    "Appointment_Weekday": 2,
    "Appointment_Month": 4,
    "Scheduled_Hour": 10,
    "Gender_M": 0,
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');
    :root { --ink:#17365f; --muted:#627692; --blue:#3568f5; --line:#dce7f4; --canvas:#f4f8fd; --sidebar:#102641; --sidebar-muted:#9fb2ca; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--ink); }
    .stApp { background:var(--canvas); }
    [data-testid="stSidebar"] { width:17rem !important; min-width:17rem !important; background:var(--sidebar); border-right:1px solid #263d58; }
    [data-testid="stSidebar"] > div:first-child { background:linear-gradient(180deg,#102641 0%,#132f50 100%); }
    [data-testid="stSidebar"] * { color:#eaf2ff; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#bdcce2; }
    [data-testid="stSidebar"] hr { border-color:#2b4562; margin:.9rem 0; }
    [data-testid="stSidebar"] [data-testid="stButton"] { margin:.12rem 0; }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        justify-content:flex-start; min-height:44px; padding:.55rem .8rem;
        border:1px solid transparent; border-radius:10px; text-align:left;
        font-size:.86rem; font-weight:600; transition:background .15s ease,border-color .15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[data-testid="stBaseButton-secondary"] {
        background:transparent; color:#d5e2f2;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[data-testid="stBaseButton-secondary"]:hover {
        background:#1b3a5c; border-color:#2b4d70; color:#fff;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[data-testid="stBaseButton-primary"] {
        background:linear-gradient(100deg,#244c77,#1b3d63); border-color:#37638f;
        color:#fff; box-shadow:inset 3px 0 #72d3ca,0 5px 14px #06182d45;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:focus-visible {
        outline:2px solid #72d3ca; outline-offset:2px;
    }
    .block-container { max-width:1480px; padding-top:1.8rem; padding-bottom:3rem; }
    .brand-lockup { display:flex; align-items:center; gap:.7rem; margin:.25rem 0 .35rem; }
    .brand-mark { width:38px; height:38px; display:grid; place-items:center; border-radius:11px; background:linear-gradient(145deg,#5b8cff,#3d66df); color:#fff; font:800 .85rem Manrope,sans-serif; box-shadow:0 5px 14px #07182e55; }
    .brand-name { color:#fff; font:800 1rem Manrope,sans-serif; letter-spacing:-.02em; }
    .brand-caption { color:#9fb2ca; font-size:.75rem; margin:0 0 1.15rem 2.95rem; }
    .sidebar-section { color:#89a1be; font-size:.65rem; font-weight:800; letter-spacing:.12em; margin:.75rem 0 .45rem; }
    .model-status { background:#193654; border:1px solid #2a496b; border-radius:12px; padding:.8rem .85rem; margin:.2rem 0 .75rem; }
    .model-status-title { display:flex; align-items:center; gap:.45rem; color:#fff; font-size:.82rem; font-weight:700; }
    .status-dot { width:8px; height:8px; border-radius:50%; background:#4bd0aa; box-shadow:0 0 0 3px #4bd0aa22; }
    .model-status-copy { color:#afc2d9; font-size:.73rem; line-height:1.5; margin:.4rem 0 0; }
    .sidebar-note { color:#93a9c2; font-size:.72rem; line-height:1.55; }
    .eyebrow { color:#3568f5; font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; }
    .page-title { font:800 1.8rem Manrope,sans-serif; color:#17365f; margin:.15rem 0 .3rem; }
    .page-subtitle { color:#627692; font-size:.92rem; margin:0 0 1.25rem; }
    .hero { background:linear-gradient(110deg,#fff 0%,#edf5ff 100%); border:1px solid #dce8f5; border-radius:16px; padding:2rem; min-height:220px; display:flex; align-items:center; box-shadow:0 12px 30px #25456a0b; }
    .hero h1 { font:800 2.15rem Manrope,sans-serif; color:#173b71; margin:.35rem 0 .5rem; }
    .hero p { max-width:620px; color:#58708f; font-size:.98rem; line-height:1.65; margin:0; }
    .hero-art { font-size:5rem; text-align:center; }
    .panel { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1.1rem 1.2rem; box-shadow:0 7px 22px #25456a09; }
    .panel-title { font:700 1rem Manrope,sans-serif; color:#17365f; margin:0 0 .25rem; }
    .panel-copy { color:#71839b; font-size:.82rem; margin:0 0 .8rem; }
    .feature-card { height:100%; background:#fff; border:1px solid var(--line); border-radius:12px; padding:1rem; box-shadow:0 5px 16px #25456a07; }
    .feature-icon { width:32px; height:32px; display:grid; place-items:center; border-radius:7px; margin-bottom:.65rem; background:#eaf1ff; color:#3568f5; font-size:1rem; }
    .feature-card strong { display:block; color:#17365f; font-size:.9rem; margin-bottom:.25rem; }
    .feature-card span { color:#71839b; font-size:.78rem; line-height:1.45; }
    .risk-panel { background:#f4f8fd; border:1px solid #dce7f4; border-radius:8px; padding:1.15rem; text-align:center; }
    .risk-panel.risk-above { background:#fff7f8; border-color:#f6dce1; }
    .risk-panel.risk-below { background:#f1faf7; border-color:#d1eee5; }
    .risk-panel.risk-high { background:#fff7f8; border-color:#f6dce1; }
    .risk-panel.risk-medium { background:#fffaf0; border-color:#f2e3bf; }
    .risk-panel.risk-low { background:#f1faf7; border-color:#d1eee5; }
    .risk-big { font:800 2.3rem Manrope,sans-serif; color:#e24a5a; margin:.25rem 0; }
    .risk-below .risk-big { color:#14856f; }
    .risk-low .risk-big { color:#14856f; }
    .risk-medium .risk-big { color:#a66b0b; }
    .small-label { font-size:.72rem; font-weight:700; color:#7e8ea4; text-transform:uppercase; }
    .stButton > button { border-radius:9px; min-height:42px; background:#3568f5; border:1px solid #3568f5; color:white; font-weight:700; transition:transform .15s ease,box-shadow .15s ease; }
    .stButton > button:hover { background:#2455db; border-color:#2455db; color:white; }
    .stButton > button:hover { transform:translateY(-1px); box-shadow:0 7px 16px #2455db30; }
    .stSelectbox > div > div, .stNumberInput input, .stSlider { border-radius:9px; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:.9rem 1rem; box-shadow:0 5px 16px #25456a07; }
    [data-testid="stMetricLabel"] { color:#71839b; }
    .footer { color:#8090a5; border-top:1px solid var(--line); margin-top:2rem; padding-top:1rem; text-align:center; font-size:.76rem; }
    @media (max-width:900px) { .block-container { padding-left:1.25rem; padding-right:1.25rem; } }
    @media (max-width:700px) { .hero { padding:1.3rem; } .hero h1 { font-size:1.7rem; } .hero-art { font-size:3rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def make_input(values):
    missing = [name for name in feature_names if name not in values]
    if missing:
        raise ValueError(f"Prediction input is missing features: {', '.join(missing)}")
    return pd.DataFrame([{name: values[name] for name in feature_names}], columns=feature_names)


def predict_probability(values):
    return float(model.predict_proba(make_input(values))[0][1])


def get_base_estimator(estimator):
    while hasattr(estimator, "estimator"):
        estimator = estimator.estimator
    return estimator


@st.cache_data
def load_appointments():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Historical appointments file is missing: {DATA_PATH}")
    return pd.read_csv(DATA_PATH, usecols=["Age", "Gender", "AppointmentDay", "No_show"])


@st.cache_data
def load_reference_features():
    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Model reference dataset is missing: {REFERENCE_DATA_PATH}")
    raw = pd.read_csv(REFERENCE_DATA_PATH)
    from src.modeling import prepare_training_data

    features, _, _, _, _ = prepare_training_data(raw)
    return features


def calculate_explanations(feature_frame):
    if shap_explainer is None:
        return None
    return shap_explainer(feature_frame)


def record_prediction(report, source):
    summary = batch_summary(report, source, MODEL_VERSION)
    st.session_state.prediction_history.insert(0, summary)
    st.session_state.prediction_history = st.session_state.prediction_history[:500]
    st.session_state.latest_report = report
    if get_connection_url():
        try:
            record_event(
                model_version=MODEL_VERSION,
                prediction_source=source,
                total_count=summary["Total"],
                low_count=summary["Low"],
                medium_count=summary["Medium"],
                high_count=summary["High"],
            )
        except (RuntimeError, ValueError, OSError, psycopg.Error) as error:
            st.error(
                "Predictions were generated, but the PostgreSQL metadata history could not "
                f"be saved: {error}"
            )
            logger.exception("Could not write aggregate prediction metadata.")


def render_report_download(report, prefix):
    csv_bytes = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prediction and explanation report (CSV)",
        data=csv_bytes,
        file_name=f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        width="stretch",
    )


def render_hospital_dashboard():
    page_heading(
        "Operations overview",
        "Hospital dashboard",
        "Historical attendance from the sample dataset and aggregate predictions from this session.",
    )
    try:
        appointments = load_appointments()
    except FileNotFoundError as error:
        st.error(str(error))
        appointments = pd.DataFrame()

    events = list(st.session_state.prediction_history)
    if get_connection_url():
        try:
            events = recent_events()
            st.caption(
                "Prediction history is loaded from PostgreSQL (aggregate counts only; "
                "last 500 runs within 90 days)."
            )
        except (RuntimeError, ValueError, OSError, psycopg.Error) as error:
            st.error(f"Could not load PostgreSQL prediction history: {error}")
            logger.exception("Could not read aggregate prediction metadata.")
    else:
        st.caption("Prediction history is in memory for this session only.")

    total_historical = len(appointments)
    historical_no_show_rate = (
        float(appointments["No_show"].mean()) if total_historical else None
    )
    predicted_total = sum(event["Total"] for event in events)
    predicted_high = sum(event["High"] for event in events)
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Appointments in sample",
        f"{total_historical:,}" if total_historical else "--",
    )
    metric_columns[1].metric(
        "Historical attendance",
        f"{1 - historical_no_show_rate:.1%}" if historical_no_show_rate is not None else "--",
    )
    metric_columns[2].metric("Appointments scored", f"{predicted_total:,}")
    metric_columns[3].metric("Predicted high risk", f"{predicted_high:,}")

    trend_column, distribution_column = st.columns([1.5, 1], gap="large")
    with trend_column:
        st.markdown(
            '<div class="panel-title">Attendance trend</div>'
            '<div class="panel-copy">Observed sample-dataset attendance by appointment date; not prospective results.</div>',
            unsafe_allow_html=True,
        )
        if total_historical:
            appointments["AppointmentDay"] = pd.to_datetime(
                appointments["AppointmentDay"], errors="coerce", utc=True
            )
            appointments = appointments.dropna(subset=["AppointmentDay"])
            appointments["Appointment Date"] = appointments[
                "AppointmentDay"
            ].dt.date.astype(str)
            trend = (
                appointments.groupby("Appointment Date", as_index=False)
                .agg(
                    Attendance=("No_show", lambda values: 1 - values.mean()),
                    Appointments=("No_show", "size"),
                )
            )
            chart = px.line(
                trend,
                x="Appointment Date",
                y="Attendance",
                markers=True,
                hover_data={"Appointments": True, "Attendance": ":.1%"},
                labels={"Attendance": "Attendance rate"},
                color_discrete_sequence=["#3568f5"],
            )
            chart.update_layout(
                height=360,
                margin=dict(l=12, r=12, t=18, b=12),
                yaxis_tickformat=".0%",
                hovermode="x unified",
            )
            st.plotly_chart(chart, width="stretch")
        else:
            st.info("Historical appointment data is not available.")
    with distribution_column:
        st.markdown(
            '<div class="panel-title">Prediction risk distribution</div>'
            '<div class="panel-copy">Session or PostgreSQL aggregate prediction counts.</div>',
            unsafe_allow_html=True,
        )
        risk_counts = {
            risk: sum(event[risk] for event in events)
            for risk in ("Low", "Medium", "High")
        }
        if sum(risk_counts.values()):
            chart = go.Figure(
                data=[
                    go.Pie(
                        labels=list(risk_counts),
                        values=list(risk_counts.values()),
                        hole=0.62,
                        sort=False,
                        marker=dict(colors=["#36a88a", "#e7aa42", "#df5f6b"]),
                        hovertemplate="%{label}: %{value} appointments (%{percent})<extra></extra>",
                    )
                ]
            )
            chart.update_layout(
                height=360,
                showlegend=True,
                margin=dict(l=8, r=8, t=18, b=8),
                legend=dict(orientation="h", y=-0.08),
                annotations=[
                    dict(
                        text=f"{sum(risk_counts.values()):,}<br>scored",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=17, color="#17365f"),
                    )
                ],
            )
            st.plotly_chart(chart, width="stretch")
        else:
            st.info("Run a single or batch prediction to populate the risk distribution.")

    st.markdown("#### Recent prediction runs")
    if events:
        st.dataframe(pd.DataFrame(events), width="stretch", hide_index=True)
    else:
        st.info("No predictions have been run yet.")
    drift = st.session_state.get("latest_drift")
    if drift is not None:
        st.markdown("#### Latest input-drift check")
        st.metric(
            "Features with high distribution shift",
            int(drift["Population Stability Index"].ge(0.25).sum()),
            help="A PSI alert is a distribution-change signal, not proof of model-performance decay.",
        )


def batch_prediction_page():
    page_heading(
        "Bulk scoring",
        "Batch prediction",
        "Upload a feature-only CSV to score multiple appointment profiles and filter the illustrative risk bands.",
    )
    st.warning(
        "Prototype is restricted to synthetic or properly de-identified examples. "
        "Do not upload real patient records, names, appointment IDs, or other identifying information."
    )
    st.markdown(
        "**Required columns:** " + ", ".join(FEATURE_NAMES),
    )
    st.caption(
        f"Maximum {MAX_BATCH_ROWS:,} rows and {MAX_UPLOAD_BYTES // (1024 * 1024)} MB per CSV. "
        "No identifiers, outcomes, or extra columns are accepted."
    )
    upload = st.file_uploader("Upload de-identified appointment profiles", type=["csv"])
    profiles = None
    if upload is not None:
        if upload.size > MAX_UPLOAD_BYTES:
            st.error(f"CSV exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
            return
        try:
            uploaded_frame = pd.read_csv(upload)
            profiles = validate_batch_frame(uploaded_frame)
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
            st.error(f"Could not accept this CSV: {error}")
            return
        st.success(f"Validated {len(profiles):,} feature-only rows. Original feature values are not saved.")
        st.dataframe(profiles.head(5), width="stretch", hide_index=True)

        with st.expander("Compare input distributions to the bundled reference dataset"):
            try:
                reference = load_reference_features()
                drift = drift_report(reference, profiles)
                st.session_state.latest_drift = drift
                st.dataframe(
                    drift.style.format({"Population Stability Index": "{:.3f}"}),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "PSI is an input-distribution alert only. It cannot establish model accuracy, "
                    "clinical validity, or performance drift; thresholds are heuristic."
                )
            except FileNotFoundError as error:
                st.error(str(error))

        confirmed = st.checkbox(
            "I confirm this file contains only synthetic or de-identified profiles and no direct identifiers."
        )
        if st.button("Score batch", type="primary", disabled=not confirmed):
            probabilities = model.predict_proba(profiles)[:, 1]
            explanation = calculate_explanations(profiles)
            report = build_prediction_report(
                probabilities,
                HIGH_RISK_THRESHOLD,
                MODEL_VERSION,
                explanation,
            )
            record_prediction(report, "batch")
            st.success(f"Scored {len(report):,} rows. Input feature values were not stored.")

    report = st.session_state.get("latest_report")
    if report is not None:
        st.markdown("#### Latest prediction report")
        risk_filter = st.multiselect(
            "Filter risk bands",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
        )
        filtered = report.loc[report["Risk Band"].isin(risk_filter)]
        counts = report["Risk Band"].value_counts()
        metrics = st.columns(4)
        metrics[0].metric("Scored", len(report))
        metrics[1].metric("Low", int(counts.get("Low", 0)))
        metrics[2].metric("Medium", int(counts.get("Medium", 0)))
        metrics[3].metric("High", int(counts.get("High", 0)))
        st.dataframe(filtered, width="stretch", hide_index=True)
        st.caption(
            f"Low <10%; medium >=10% and <{HIGH_RISK_THRESHOLD:.1%}; "
            f"high >={HIGH_RISK_THRESHOLD:.1%}. Thresholds are illustrative."
        )
        render_report_download(report, "carepredict_batch_report")


def external_validation_page():
    page_heading(
        "Model monitoring",
        "Data & validation",
        "Check feature-distribution shift or evaluate outcomes from a separate authorized, de-identified dataset.",
    )
    st.warning(
        "Use synthetic/de-identified data only in this prototype. Do not upload identifiable or real patient records. "
        "External validation outcomes are evaluated in memory and are not persisted."
    )
    st.markdown(
        "**Required:** the 12 model feature columns plus a `No_show` outcome column "
        "(0 = attended, 1 = no-show). No identifiers or additional fields."
    )
    upload = st.file_uploader(
        "Upload labeled validation profiles",
        type=["csv"],
        key="external_validation_upload",
    )
    if upload is None:
        return
    if upload.size > MAX_UPLOAD_BYTES:
        st.error(f"CSV exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
        return
    try:
        frame = pd.read_csv(upload)
        if "No_show" not in frame.columns:
            raise ValueError("The required No_show outcome column is missing.")
        outcomes = pd.to_numeric(frame.pop("No_show"), errors="coerce")
        profiles = validate_batch_frame(frame)
        if outcomes.isna().any() or not outcomes.isin([0, 1]).all():
            raise ValueError("No_show must contain only 0 (attended) and 1 (no-show).")
        if outcomes.nunique() != 2:
            raise ValueError("Validation requires examples of both attended and no-show outcomes.")
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        st.error(f"Could not accept this validation CSV: {error}")
        return
    confirmed = st.checkbox(
        "I confirm this file is authorized, synthetic or de-identified, and contains no identifiers."
    )
    if st.button("Evaluate on uploaded outcomes", type="primary", disabled=not confirmed):
        probabilities = model.predict_proba(profiles)[:, 1]
        result = pd.DataFrame(
            {
                "ROC-AUC": [roc_auc_score(outcomes, probabilities)],
                "Average precision": [average_precision_score(outcomes, probabilities)],
                "Brier score": [brier_score_loss(outcomes, probabilities)],
                "Rows": [len(outcomes)],
                "No-show rate": [float(outcomes.mean())],
            }
        )
        st.session_state.external_validation_report = result
        st.dataframe(result, width="stretch", hide_index=True)
        st.info(
            "This is an external dataset check, not a prospective impact study. "
            "The public sample and this prototype do not establish clinical usefulness."
        )
        st.download_button(
            "Download external validation metrics (CSV)",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name=f"carepredict_external_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


def page_heading(eyebrow, title, subtitle):
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div><div class="page-title">{title}</div><div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def navigate_to(page):
    st.session_state.navigation = page


def prediction_page():
    page_heading("Risk screening", "Predict Patient No-Show Risk", "Enter the patient and appointment details to estimate attendance risk.")
    form_col, result_col = st.columns([1.65, 1], gap="large")

    with form_col:
        st.markdown('<div class="panel-title">Appointment Details</div>', unsafe_allow_html=True)
        with st.form("prediction_form"):
            left, right = st.columns(2)
            with left:
                age = st.number_input("Age", min_value=0, max_value=120, value=30)
                gender = st.selectbox("Gender", ["Female", "Male"])
                scholarship = st.selectbox("Scholarship", ["No", "Yes"])
                hypertension = st.selectbox("Hypertension", ["No", "Yes"])
                diabetes = st.selectbox("Diabetes", ["No", "Yes"])
                alcoholism = st.selectbox("Alcoholism", ["No", "Yes"])
            with right:
                handicap = st.selectbox("Handicap level", [0, 1, 2, 3, 4])
                sms = st.selectbox("SMS Reminder", ["Yes", "No"])
                days = st.number_input("Days in Advance", min_value=0, max_value=365, value=7)
                weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                weekday = st.selectbox("Appointment Weekday", range(7), format_func=lambda value: weekday_names[value], index=2)
                month = st.selectbox("Appointment Month", range(1, 13), index=3, format_func=lambda value: datetime(2024, value, 1).strftime("%B"))
                hour = st.slider("Scheduled Hour", 0, 23, 10)
            submitted = st.form_submit_button("Predict Risk", width="stretch")

        if submitted:
            values = {
                "Age": age,
                "Scholarship": int(scholarship == "Yes"),
                "Hypertension": int(hypertension == "Yes"),
                "Diabetes": int(diabetes == "Yes"),
                "Alcoholism": int(alcoholism == "Yes"),
                "Handicap": handicap,
                "SMS_received": int(sms == "Yes"),
                "Days_In_Advance": days,
                "Appointment_Weekday": weekday,
                "Appointment_Month": month,
                "Scheduled_Hour": hour,
                "Gender_M": int(gender == "Male"),
            }
            probability = predict_probability(values)
            risk_level = classify_risk([probability], HIGH_RISK_THRESHOLD)[0]
            st.session_state.current_input = values
            st.session_state.current_probability = probability
            explanation = calculate_explanations(make_input(values))
            report = build_prediction_report(
                [probability],
                HIGH_RISK_THRESHOLD,
                MODEL_VERSION,
                explanation,
            )
            record_prediction(report, "single")

    with result_col:
        st.markdown('<div class="panel-title">Predicted No-Show Risk</div>', unsafe_allow_html=True)
        probability = st.session_state.get("current_probability")
        if probability is None:
            st.markdown('<div class="risk-panel"><div class="small-label">Ready for screening</div><div class="risk-big">--</div><div class="panel-copy">Complete the patient details and predict risk.</div></div>', unsafe_allow_html=True)
        else:
            level = classify_risk([probability], HIGH_RISK_THRESHOLD)[0]
            risk_class = f"risk-{level.casefold()}"
            st.markdown(f'<div class="risk-panel {risk_class}"><div class="small-label">Example outreach screening flag</div><div class="risk-big">{level}</div><div style="font-size:2rem;font-weight:800;color:#17365f">{probability:.0%}</div><div class="panel-copy">Calibrated model estimate of a missed appointment</div></div>', unsafe_allow_html=True)
            st.progress(probability, text="Probability of no-show")
            st.info(
                f"**Estimated no-show probability: {probability:.0%} ({level} risk).** "
                "This is a model estimate, not a certainty or care recommendation. "
                f"Risk bands: low <10%, medium >=10% and <{HIGH_RISK_THRESHOLD:.1%}, "
                f"high >={HIGH_RISK_THRESHOLD:.1%}. The demo threshold is illustrative "
                "and has not been clinically validated."
            )
            if st.session_state.get("latest_report") is not None:
                render_report_download(st.session_state.latest_report, "carepredict_prediction")
        st.caption("Not a clinical diagnosis.")


def explainability_page():
    page_heading("Model transparency", "Prediction Explanation (SHAP)", "Review the strongest model drivers and their impact on the selected patient.")
    values = st.session_state.get("current_input", DEFAULT_INPUT)
    input_frame = make_input(values)
    importance = getattr(
        get_base_estimator(model), "feature_importances_", np.zeros(len(feature_names))
    )
    global_frame = pd.DataFrame({"Feature": [FEATURE_LABELS[name] for name in feature_names], "Importance": importance})
    global_frame = global_frame.sort_values("Importance", ascending=False).set_index("Feature")

    global_col, local_col = st.columns(2, gap="medium")
    with global_col:
        st.markdown('<div class="panel-title">Feature Importance</div><div class="panel-copy">Global model importance, ranked by contribution.</div>', unsafe_allow_html=True)
        st.bar_chart(global_frame.head(8), horizontal=True, color="#4f7cf7")
    with local_col:
        st.markdown('<div class="panel-title">Patient-Level Explanation</div><div class="panel-copy">Feature impact for the current patient profile.</div>', unsafe_allow_html=True)
        if shap_explainer is not None:
            try:
                explanation = shap_explainer(input_frame)
                shap_values = np.asarray(explanation.values)
                if shap_values.ndim == 3:
                    shap_values = shap_values[0, :, -1]
                elif shap_values.ndim == 2:
                    shap_values = shap_values[0]
                local = pd.DataFrame({"Feature": [FEATURE_LABELS[name] for name in feature_names], "SHAP value": shap_values})
                local = local.reindex(local["SHAP value"].abs().sort_values(ascending=False).index).set_index("Feature")
                st.bar_chart(local.head(8), horizontal=True, color="#08a89d")
                st.dataframe(local.head(8), width="stretch")
            except Exception as error:
                st.warning(f"The saved SHAP explainer could not calculate this profile: {error}")
                st.caption("Showing native XGBoost feature importance instead.")
                st.bar_chart(global_frame.head(8), horizontal=True, color="#08a89d")
        else:
            st.info(f"{shap_error} Global XGBoost importance is shown here instead.")
            st.bar_chart(global_frame.head(8), horizontal=True, color="#08a89d")

    st.markdown('<div class="panel"><div class="panel-title">Interpretation</div><div class="panel-copy">SHAP explains the underlying tree model score before probability calibration. Feature importance describes learned associations, not causes; neither replaces human judgment.</div></div>', unsafe_allow_html=True)


def what_if_page():
    page_heading("Scenario testing", "What-If Analysis", "Compare model estimates for alternate input values; this is not a causal intervention estimate.")
    base = dict(st.session_state.get("current_input", DEFAULT_INPUT))
    choices = {"Days in Advance": ("Days_In_Advance", 0, 30), "Age": ("Age", 0, 100), "SMS Reminder": ("SMS_received", 0, 1)}
    setup, chart = st.columns([1, 1.4], gap="large")
    with setup:
        st.markdown('<div class="panel-title">Select a Feature to Modify</div>', unsafe_allow_html=True)
        label = st.selectbox("Feature", list(choices))
        key, minimum, maximum = choices[label]
        current = int(base[key])
        if key == "SMS_received":
            current = st.selectbox("Current Value", [0, 1], index=current, format_func=lambda value: "Yes" if value else "No")
            tested_value = 1 - current
            st.caption(f"Compare reminder SMS: {'Yes' if tested_value else 'No'}")
        else:
            current = st.slider("Current Value", minimum, maximum, min(current, maximum))
            tested_value = st.slider("Test Value", minimum, maximum, min(maximum, current + 5))
        baseline_input = dict(base)
        baseline_input[key] = current
        baseline = predict_probability(baseline_input)
        changed = dict(baseline_input)
        changed[key] = tested_value
        changed_probability = predict_probability(changed)
        st.metric("Risk Change", f"{changed_probability - baseline:+.1%}", help="Change in estimated no-show probability after the selected adjustment.")
        st.caption(f"Baseline {baseline:.1%}  ·  Scenario {changed_probability:.1%}")

    with chart:
        st.markdown('<div class="panel-title">Risk Across Different Values</div><div class="panel-copy">Model-estimated probability as the selected feature changes.</div>', unsafe_allow_html=True)
        if key == "SMS_received":
            candidate_values = [0, 1]
        else:
            candidate_values = list(range(minimum, maximum + 1))
        scenario_rows = []
        for candidate in candidate_values:
            scenario = dict(base)
            scenario[key] = current
            scenario[key] = candidate
            scenario_rows.append({name: scenario[name] for name in feature_names})
        scenario_frame = pd.DataFrame(scenario_rows, columns=feature_names)
        probabilities = model.predict_proba(scenario_frame)[:, 1]
        chart_data = pd.DataFrame({"No-show probability": probabilities}, index=candidate_values)
        chart_data.index.name = label
        st.line_chart(chart_data, color="#4f7cf7", height=330)
        st.caption(
            f"Selected scenario estimate: {changed_probability:.1%}. "
            "Changing an input here does not establish that changing it in real life would cause this risk change."
        )


def history_page():
    page_heading(
        "Recent activity",
        "Prediction History & Reports",
        "Session history and optional PostgreSQL aggregate audit records. Uploaded features and identifiers are not stored.",
    )
    history = list(st.session_state.prediction_history)
    if get_connection_url():
        try:
            history = recent_events()
            st.caption("Showing persisted aggregate metadata from PostgreSQL.")
        except (RuntimeError, ValueError, OSError, psycopg.Error) as error:
            st.error(f"Could not load PostgreSQL prediction history: {error}")
            logger.exception("Could not read aggregate prediction metadata.")
    else:
        st.caption("PostgreSQL is not configured. This session's aggregate history is temporary.")
    if history:
        st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)
        if not get_connection_url() and st.button(
            "Clear current-session history", type="secondary"
        ):
            st.session_state.prediction_history = []
            st.session_state.latest_report = None
            st.rerun()
        elif get_connection_url():
            st.caption("PostgreSQL audit summaries expire after 90 days and cannot be cleared from this page.")
    else:
        st.info("No prediction runs yet. Run a single or batch prediction to create a history entry.")
    report = st.session_state.get("latest_report")
    if report is not None:
        st.markdown("#### Latest report (current session only)")
        st.dataframe(report, width="stretch", hide_index=True)
        render_report_download(report, "carepredict_report")
        if st.button("Clear current-session report", type="secondary"):
            st.session_state.latest_report = None
            st.rerun()
    st.caption(
        "PostgreSQL stores only run timestamp, model version, source, and aggregate risk counts. "
        "Per-row results remain in this session and are not written to the database."
    )


def about_page():
    page_heading("Project overview", "About CarePredict-XAI", "Explainable machine learning for appointment attendance support.")
    about, details = st.columns([1.25, 1], gap="medium")
    with about:
        st.markdown('<div class="panel"><div class="feature-icon">♥</div><div class="panel-title">CarePredict-XAI</div><p>Explainable appointment no-show prediction prototype.</p><p class="panel-copy">This project uses a public historical dataset to explore model estimates and explanations. It does not manage appointments, connect to a clinic schedule, or recommend interventions.</p></div>', unsafe_allow_html=True)
    with details:
        st.markdown('<div class="panel-title">Key Features</div>', unsafe_allow_html=True)
        st.markdown(
            "- Historical attendance dashboard\n"
            "- Single and batch no-show estimates\n"
            "- SHAP explanations and what-if comparisons\n"
            "- CSV reports and prediction history\n"
            "- Input-distribution monitoring and external validation"
        )
        st.markdown('<div class="panel-title">Model</div>', unsafe_allow_html=True)
        st.write(
            f"Calibrated XGBoost · patient-grouped holdout ROC-AUC "
            f"{metadata['patient_grouped_test']['roc_auc']:.3f}"
        )
        st.caption(
            f"Selected using validation average precision; illustrative screening threshold "
            f"{HIGH_RISK_THRESHOLD:.1%} targets {TARGET_RECALL:.0%} recall on validation data."
        )
    st.caption(
        "Dataset: Medical Appointment No Shows (Kaggle). No independently sourced "
        "facility dataset or prospective intervention outcomes have been provided."
    )
    st.warning(
        "Prototype only: do not enter identifiable patient data or use this application "
        "to make or automate care decisions. It has not been clinically validated."
    )


st.session_state.setdefault("prediction_history", [])
st.session_state.setdefault("latest_report", None)
st.session_state.setdefault("latest_drift", None)

navigation_items = [
    ("Hospital Dashboard", "⌂  Hospital Dashboard"),
    ("Single Prediction", "▣  Single Prediction"),
    ("Batch Prediction", "▤  Batch Prediction"),
    ("Explainability (SHAP)", "⌁  Explainability (SHAP)"),
    ("What-If Analysis", "↗  What-If Analysis"),
    ("History & Reports", "▧  History & Reports"),
    ("Data & Validation", "◫  Data & Validation"),
    ("About", "ⓘ  About"),
]
navigation_destinations = {destination for destination, _ in navigation_items}
if st.session_state.get("navigation") not in navigation_destinations:
    st.session_state.navigation = "Hospital Dashboard"

with st.sidebar:
    st.markdown(
        '<div class="brand-lockup"><div class="brand-mark">CP</div>'
        '<div class="brand-name">CarePredict-XAI</div></div>'
        '<div class="brand-caption">No-show model explorer</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-section">WORKSPACE</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="sidebar-section">NAVIGATION</div>', unsafe_allow_html=True)
    page = st.session_state.navigation
    for destination, label in navigation_items:
        st.button(
            label,
            key=f"navigation_{destination}",
            type="primary" if page == destination else "secondary",
            width="stretch",
            on_click=navigate_to,
            args=(destination,),
        )
    st.divider()
    st.markdown(
        '<div class="sidebar-section">MODEL STATUS</div>'
        '<div class="model-status"><div class="model-status-title">'
        '<span class="status-dot"></span> Model ready</div>'
        '<div class="model-status-copy">Calibrated XGBoost<br>'
        f"Patient-grouped ROC-AUC · {metadata['patient_grouped_test']['roc_auc']:.3f}"
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-note">Local-only prototype · synthetic or de-identified data only.<br>'
        f"Prediction history: {'PostgreSQL aggregates' if get_connection_url() else 'session only'}."
        '</div>',
        unsafe_allow_html=True,
    )

if page == "Hospital Dashboard":
    render_hospital_dashboard()
elif page == "Single Prediction":
    prediction_page()
elif page == "Batch Prediction":
    batch_prediction_page()
elif page == "Explainability (SHAP)":
    explainability_page()
elif page == "What-If Analysis":
    what_if_page()
elif page == "History & Reports":
    history_page()
elif page == "Data & Validation":
    external_validation_page()
else:
    about_page()

st.markdown('<div class="footer">CarePredict-XAI · Explainable appointment risk dashboard</div>', unsafe_allow_html=True)
