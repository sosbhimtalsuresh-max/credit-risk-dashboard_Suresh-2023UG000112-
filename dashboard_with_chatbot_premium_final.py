"""
CredLens Pro – Credit Risk Intelligence Platform
Bug-fixed · Large fonts · Warm professional color scheme
"""

import os, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CAT_COLS = [
    "person_home_ownership", "loan_intent",
    "loan_grade", "cb_person_default_on_file",
]
LABEL_MAPS = {
    "person_home_ownership":     ["MORTGAGE", "OTHER", "OWN", "RENT"],
    "loan_intent":               ["DEBTCONSOLIDATION", "EDUCATION",
                                   "HOMEIMPROVEMENT", "MEDICAL",
                                   "PERSONAL", "VENTURE"],
    "loan_grade":                ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["N", "Y"],
}
FEAT_COLS = [
    "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length",
    "person_home_ownership_enc", "loan_intent_enc",
    "loan_grade_enc", "cb_person_default_on_file_enc",
]
NUMERIC_FEATS = [
    "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length",
]
LOG_FILE = "prediction_log.csv"
LOG_COLS = [
    "timestamp", "institution", "person_age", "person_income",
    "person_home_ownership", "person_emp_length", "loan_intent",
    "loan_grade", "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_default_on_file", "cb_person_cred_hist_length",
    "rf_probability_pct", "lr_probability_pct", "predicted_default",
]

# ── Colour palette (warm, high contrast, professional) ──
C_RED    = "#D32F2F"
C_GREEN  = "#2E7D32"
C_AMBER  = "#E65100"
C_TEAL   = "#00695C"
C_INDIGO = "#283593"
C_PURPLE = "#6A1B9A"
C_SLATE  = "#37474F"

GRADE_COLORS = {
    "A": "#2E7D32", "B": "#558B2F", "C": "#F9A825",
    "D": "#E65100", "E": "#C62828", "F": "#880E4F", "G": "#4A148C",
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CredLens Pro · Credit Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — warm light theme, big readable fonts, NO dark strains
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --bg: #F7F5F2;
    --card: #FFFFFF;
    --text: #172033;
    --muted: #6B7280;
    --line: #E8E2DA;
    --navy: #16213E;
    --navy-2: #0F172A;
    --blue: #2D5BFF;
    --green: #238636;
    --orange: #E85D04;
    --red: #D62828;
    --purple: #7B2CBF;
}

html, body, [class*="css"], .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Segoe UI', Inter, Arial, sans-serif !important;
    font-size: 16px !important;
}

.block-container {
    max-width: 1440px !important;
    padding: 1.55rem 2.25rem 2.75rem !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy) 0%, var(--navy-2) 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.04) !important;
}
section[data-testid="stSidebar"] * {
    color: #EEF2FF !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.96rem !important;
    font-weight: 600 !important;
    padding: 0.26rem 0 !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    font-size: 0.92rem !important;
}

h1 {
    font-size: 2.45rem !important;
    font-weight: 900 !important;
    color: #0F172A !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 0.15rem !important;
}
h2 {
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    color: #14213D !important;
}
h3 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #14213D !important;
}
p {
    font-size: 0.98rem !important;
    color: #445063 !important;
}

.kpi-box {
    background: linear-gradient(180deg, #FFFFFF 0%, #FCFCFD 100%);
    border-radius: 18px;
    padding: 1.05rem 1.1rem 0.95rem;
    border: 1px solid #ECE7E0;
    border-left: 5px solid;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    min-width: 0 !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.kpi-box:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
}
.kpi-label {
    font-size: 0.70rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #7B8593 !important;
    margin-bottom: 0.45rem;
}
.kpi-num {
    font-size: 2.05rem !important;
    font-weight: 900 !important;
    line-height: 1.02 !important;
    letter-spacing: -0.035em !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.kpi-sub {
    font-size: 0.84rem !important;
    color: #7B8593 !important;
    margin-top: 0.28rem;
    font-weight: 600;
}

.sec-lbl {
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #5F6B7A !important;
    border-bottom: 1px solid var(--line);
    padding-bottom: 0.42rem;
    margin-bottom: 0.85rem;
}

[data-testid="stTabs"] button {
    font-size: 0.96rem !important;
    font-weight: 700 !important;
    color: #5B6472 !important;
    padding: 0.55rem 1rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #0F172A !important;
    border-bottom: 3px solid var(--red) !important;
}

label,
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label,
[data-testid="stSlider"] label,
[data-testid="stRadio"] > label,
[data-testid="stMultiSelect"] label {
    font-size: 0.97rem !important;
    font-weight: 700 !important;
    color: #162033 !important;
}
input, select, textarea {
    font-size: 0.96rem !important;
    color: #162033 !important;
}

[data-testid="stExpander"] summary {
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #1E293B !important;
}

[data-testid="stDataFrame"] {
    font-size: 0.93rem !important;
}
[data-testid="stDataFrame"] th {
    font-size: 0.92rem !important;
    font-weight: 800 !important;
    color: #1F2937 !important;
}
[data-testid="stDataFrame"] td {
    font-size: 0.91rem !important;
    color: #334155 !important;
}

[data-testid="stCaptionContainer"], .stCaption {
    font-size: 0.90rem !important;
    color: #6B7280 !important;
    font-weight: 500 !important;
}
[data-testid="stAlert"] p, [data-testid="stAlert"] div {
    font-size: 0.96rem !important;
    font-weight: 500 !important;
}

[data-testid="stFormSubmitButton"] button, .stButton button {
    font-size: 0.98rem !important;
    font-weight: 800 !important;
    border-radius: 12px !important;
    padding: 0.72rem 1.6rem !important;
    background: linear-gradient(135deg, var(--navy), #213A7A) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 8px 22px rgba(26, 35, 64, 0.22) !important;
    letter-spacing: 0.01em;
}
[data-testid="stFormSubmitButton"] button:hover, .stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 12px 28px rgba(26, 35, 64, 0.28) !important;
}

hr {
    border-color: var(--line) !important;
    margin: 1rem 0 !important;
}

[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background-color: #1A2340 !important;
    color: #FFFFFF !important;
    font-size: 0.86rem !important;
    border-radius: 999px !important;
}

div[data-testid="stMetric"] {
    background: #fff;
    border-radius: 14px;
}

@media (max-width: 1200px) {
    .block-container {
        padding: 1.15rem 1rem 2rem !important;
    }
    h1 {
        font-size: 2rem !important;
    }
    .kpi-num {
        font-size: 1.7rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def kpi(label, value, sub="", color=C_INDIGO):
    st.markdown(f"""
    <div class="kpi-box" style="border-left-color:{color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-num" style="color:{color};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def sec(text):
    st.markdown(f'<p class="sec-lbl">{text}</p>', unsafe_allow_html=True)


def interpretation_box(title, bullets, tone="neutral"):
    palette = {
        "neutral": ("#FFFDF7", "#E0C97F", "#6B5A1E"),
        "good": ("#F2FBF3", "#81C784", "#1B5E20"),
        "warn": ("#FFF6F3", "#FFAB91", "#8D3B12"),
        "info": ("#F4F7FF", "#90CAF9", "#123A72"),
    }
    bg, bd, tx = palette.get(tone, palette["neutral"])
    bullet_html = "".join([f"<li style='margin:0.38rem 0;'>{b}</li>" for b in bullets])
    st.markdown(
        f"""
        <div style="background:{bg};border-left:6px solid {bd};padding:1rem 1.25rem;
                    border-radius:14px;margin:0.9rem 0 1.25rem 0;">
            <div style="font-size:1.15rem;font-weight:800;color:{tx};margin-bottom:0.55rem;">{title}</div>
            <ul style="margin:0 0 0 1.1rem;padding:0;font-size:1.04rem;color:#2F2F2F;">{bullet_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── KEY FIX: light_layout() returns only paper/plot/font/margin.
#    NEVER includes xaxis or yaxis — those are always passed separately.
def chart_base(height=400):
    """Base layout dict — NO xaxis/yaxis to avoid duplicate-key errors."""
    return dict(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FBFAF8",
        font=dict(family="Segoe UI, Arial", size=14, color="#1C1C1C"),
        margin=dict(t=52, b=54, l=28, r=24),
        legend=dict(font=dict(size=12, color="#1C1C1C"),
                    bgcolor="rgba(0,0,0,0)"),
        height=height,
    )


AXIS_STYLE = dict(
    gridcolor="#EEE9E2",
    linecolor="#CFC8BE",
    tickfont=dict(size=12, color="#374151"),
    title_font=dict(size=13, color="#1C1C1C"),
    showgrid=True,
)


def apply_axes(fig, xtitle="", ytitle="", xrange=None, yrange=None):
    """Apply axis titles and style — never conflicts with chart_base()."""
    xu = dict(**AXIS_STYLE, title=xtitle)
    yu = dict(**AXIS_STYLE, title=ytitle)
    if xrange:
        xu["range"] = xrange
    if yrange:
        yu["range"] = yrange
    fig.update_xaxes(**xu)
    fig.update_yaxes(**yu)
    return fig


def encode_for_model(row):
    vec = [float(row[k]) for k in [
        "person_age", "person_income", "person_emp_length",
        "loan_amnt", "loan_int_rate", "loan_percent_income",
        "cb_person_cred_hist_length",
    ]]
    for cat in CAT_COLS:
        vec.append(float(LABEL_MAPS[cat].index(row[cat])))
    return np.array(vec).reshape(1, -1)


def load_log():
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=LOG_COLS)


def append_log(record):
    df_log = load_log()
    df_log = pd.concat([df_log, pd.DataFrame([record])], ignore_index=True)
    df_log.to_csv(LOG_FILE, index=False)


def interpret_default_rate(rate):
    if rate >= 30:
        return "very elevated"
    if rate >= 20:
        return "elevated"
    if rate >= 10:
        return "moderate"
    return "low"


def chatbot_response(query, df, M, log_df):
    q = query.strip().lower()
    if not q:
        return "Please type a question about portfolio risk, model performance, loan grades, intents, or prediction records."

    total = len(df)
    defaults = int(df["loan_status"].sum())
    default_rate = defaults / total * 100 if total else 0
    avg_loan = df["loan_amnt"].mean() if total else 0
    avg_rate = df["loan_int_rate"].mean() if total else 0

    if any(k in q for k in ["overview", "summary", "dataset", "portfolio"]):
        return (
            f"Portfolio overview: the cleaned dataset contains {total:,} borrowers, with {defaults:,} defaults "
            f"and an overall default rate of {default_rate:.2f}%. The average loan amount is ${avg_loan:,.0f} "
            f"and the average interest rate is {avg_rate:.2f}%."
        )

    if "auc" in q or "model" in q or "accuracy" in q or "best model" in q:
        rf_acc = M["rf_rep"]["accuracy"]
        lr_acc = M["lr_rep"]["accuracy"]
        better = "Random Forest" if M["rf_auc"] >= M["lr_auc"] else "Logistic Regression"
        return (
            f"Model performance: Random Forest AUC = {M['rf_auc']:.4f}, accuracy = {rf_acc:.4f}. "
            f"Logistic Regression AUC = {M['lr_auc']:.4f}, accuracy = {lr_acc:.4f}. "
            f"Based on AUC, {better} is the stronger classifier on this test split."
        )

    if "riskiest grade" in q or "highest default grade" in q or "which loan grade is riskiest" in q:
        grade_summary = (
            df.groupby("loan_grade", observed=True)["loan_status"]
            .mean().mul(100).sort_values(ascending=False)
        )
        g = grade_summary.index[0]
        rate = grade_summary.iloc[0]
        return f"The riskiest observed loan grade is {g}, with a default rate of {rate:.2f}% in this dataset."

    if "grade" in q and any(g.lower() in q for g in list("abcdefg")):
        for g in ["A", "B", "C", "D", "E", "F", "G"]:
            if f"grade {g.lower()}" in q or q.endswith(g.lower()) or f" {g.lower()} " in q:
                sub = df[df["loan_grade"] == g]
                if len(sub) == 0:
                    return f"No records are available for loan grade {g}."
                rate = sub["loan_status"].mean() * 100
                avg_amt = sub["loan_amnt"].mean()
                return (
                    f"Loan grade {g}: {len(sub):,} loans, default rate {rate:.2f}%, and average loan amount ${avg_amt:,.0f}. "
                    f"That risk level is {interpret_default_rate(rate)} relative to the rest of the portfolio."
                )

    if "highest default" in q or "most risky intent" in q or "loan intent" in q:
        intent_summary = (
            df.groupby("loan_intent", observed=True)["loan_status"]
            .mean().mul(100).sort_values(ascending=False)
        )
        top_intent = intent_summary.index[0]
        top_rate = intent_summary.iloc[0]
        low_intent = intent_summary.index[-1]
        low_rate = intent_summary.iloc[-1]
        return (
            f"By loan intent, the highest observed default rate is for {top_intent} at {top_rate:.2f}%. "
            f"The lowest is {low_intent} at {low_rate:.2f}%."
        )

    if "home ownership" in q or "ownership" in q:
        home_summary = (
            df.groupby("person_home_ownership", observed=True)["loan_status"]
            .mean().mul(100).sort_values(ascending=False)
        )
        top = home_summary.index[0]
        rate = home_summary.iloc[0]
        return f"Home ownership insight: {top} has the highest observed default rate at {rate:.2f}% in this dataset."

    if "feature importance" in q or "important feature" in q or "drivers" in q:
        imp = pd.DataFrame({"feature": FEAT_COLS, "importance": M["feat_imp"]}).sort_values("importance", ascending=False).head(5)
        parts = [f"{row.feature.replace('_enc','').replace('_',' ')} ({row.importance:.3f})" for row in imp.itertuples()]
        return "Top Random Forest risk drivers are: " + ", ".join(parts) + "."

    if "prediction log" in q or "footprint" in q or "records" in q:
        if log_df.empty:
            return "The Footprint Database is currently empty. New records will appear after you run predictions in the Loan Risk Predictor."
        high = int(log_df["predicted_default"].sum()) if "predicted_default" in log_df.columns else 0
        total_logs = len(log_df)
        return (
            f"The Footprint Database contains {total_logs:,} saved prediction records. "
            f"Among them, {high:,} are marked high risk and {total_logs - high:,} are marked low risk."
        )

    if "how to reduce" in q or "reduce risk" in q or "improve approval" in q:
        return (
            "To reduce predicted risk, the most practical levers in this dataset are lowering loan amount relative to income, "
            "improving credit history depth, avoiding prior default flags, and applying for lower-risk loan grades when possible. "
            "These steps improve profile quality, but they do not guarantee approval."
        )

    return (
        "I can help with dataset summary, loan grade risk, loan intent risk, home ownership patterns, model AUC/accuracy, "
        "top feature importance, and saved prediction records. Try questions like: 'Which loan grade is riskiest?', "
        "'What is the overall default rate?', or 'How many records are in the Footprint Database?'"
    )

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.4rem 0 0.6rem;'>
        <div style='font-size:3rem;'>🏦</div>
        <div style='font-size:1.5rem;font-weight:900;color:#FFFFFF;
                    letter-spacing:-0.01em;margin-top:0.4rem;'>CredLens Pro</div>
        <div style='font-size:0.9rem;color:#93B4E8;margin-top:0.3rem;
                    font-weight:500;letter-spacing:0.06em;'>
            CREDIT RISK INTELLIGENCE
        </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader("📂  Upload CSV", type=["csv"],
                                 help="Upload credit_risk_Modelling.csv")
    st.divider()

    st.markdown(
        "<p style='font-size:0.78rem;font-weight:800;letter-spacing:0.15em;"
        "color:#93B4E8;text-transform:uppercase;margin-bottom:0.4rem;'>"
        "Navigation</p>", unsafe_allow_html=True)

    NAV = [
        "📊  Overview",
        "🔍  Deep Analytics",
        "📈  Aggregations",
        "🔗  Correlation & Heatmap",
        "🤖  ML Models",
        "🎯  Loan Risk Predictor",
        "🗄️  Footprint Database",
        "💬  Risk Chatbot",
    ]
    nav = st.radio("", NAV, label_visibility="collapsed")
    st.divider()
    st.markdown("""
    <div style='text-align:center;font-size:0.88rem;color:#93B4E8;
                padding:0.4rem 0;line-height:1.8;'>
        RF AUC &nbsp;·&nbsp; <strong style='color:#FFFFFF;'>0.9823</strong><br>
        LR AUC &nbsp;·&nbsp; <strong style='color:#FFFFFF;'>0.8458</strong><br>
        <span style='font-size:0.78rem;opacity:0.6;'>v 2.2 · CredLens Pro</span>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset …")
def load_raw(src):
    if src is None:
        candidate_paths = [
            "credit_risk_Modelling.csv",
            "credit_risk_modelling_dataset.csv",
            "credit_risk_modelling_dataset.csv.csv",
            "data/credit_risk_Modelling.csv",
            "data/credit_risk_modelling_dataset.csv",
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                df = pd.read_csv(p)
                df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", case=False)]
                return df
        return None
    df = pd.read_csv(src)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", case=False)]
    return df


@st.cache_data(show_spinner="Cleaning & engineering features …")
def full_pipeline(raw):
    df = raw.copy()
    required_cols = [
        "person_age", "person_income", "person_home_ownership", "person_emp_length",
        "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_status",
        "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    me = df["person_emp_length"].mean()
    mr = df["loan_int_rate"].mean()
    df["person_emp_length"] = df["person_emp_length"].fillna(me)
    df["loan_int_rate"]     = df["loan_int_rate"].fillna(mr)
    df = df[(df["person_age"] >= 18) & (df["person_age"] <= 100)].copy()

    df["risk_segment"]     = df["loan_status"].map({1: "High Risk", 0: "Low Risk"})
    df["income_group"]     = pd.cut(
        df["person_income"], bins=[-np.inf, 29999, 69999, np.inf],
        labels=["Low Income", "Middle Income", "High Income"])
    df["employment_group"] = pd.cut(
        df["person_emp_length"], bins=[-np.inf, 1.999, 4.999, np.inf],
        labels=["New Employee", "Mid Experience", "Experienced"])
    for cat in CAT_COLS:
        classes = LABEL_MAPS[cat]
        df[cat + "_enc"] = df[cat].apply(
            lambda x: classes.index(x) if x in classes else 0)
    return df, me, mr


raw_df = load_raw(uploaded)
if raw_df is None:
    st.markdown("""
    <div style="background:#FFFFFF;border-radius:16px;padding:3.5rem;
                text-align:center;box-shadow:0 2px 16px rgba(0,0,0,0.09);
                margin-top:3rem;border-top:5px solid #D32F2F;">
        <div style="font-size:4rem;">📂</div>
        <h2 style="color:#0F1628;margin:1rem 0 0.6rem;font-size:1.8rem;">
            No Data Found</h2>
        <p style="font-size:1.1rem;color:#666;max-width:480px;margin:0 auto;">
            Upload <strong>credit_risk_Modelling.csv</strong> using the
            sidebar uploader, or place it in the same folder as
            <code>app.py</code> and restart.
        </p>
    </div>""", unsafe_allow_html=True)
    st.stop()

try:
    df, mean_emp, mean_rate = full_pipeline(raw_df)
except Exception as e:
    st.error(f"Dataset validation error: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# ML TRAINING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training ML models … (one-time, ~20 s)")
def train_models(df):
    X = df[FEAT_COLS].copy()
    y = df["loan_status"].copy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=150, max_depth=15,
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_p = rf.predict_proba(X_te)[:, 1]
    rf_pred = (rf_p >= 0.5).astype(int)

    sc = StandardScaler()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(sc.fit_transform(X_tr), y_tr)
    lr_p = lr.predict_proba(sc.transform(X_te))[:, 1]
    lr_pred = (lr_p >= 0.5).astype(int)

    return dict(
        rf=rf, lr=lr, sc=sc, y_te=y_te,
        rf_p=rf_p, rf_pred=rf_pred,
        lr_p=lr_p, lr_pred=lr_pred,
        rf_auc=round(roc_auc_score(y_te, rf_p), 4),
        lr_auc=round(roc_auc_score(y_te, lr_p), 4),
        rf_rep=classification_report(y_te, rf_pred, output_dict=True),
        lr_rep=classification_report(y_te, lr_pred, output_dict=True),
        feat_imp=list(rf.feature_importances_),
    )


M = train_models(df)

# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATIONS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def agg_all(df):
    def _a(grp):
        return (
            df.groupby(grp, observed=True)
            .agg(
                total=("loan_status", "count"),
                default_rate=("loan_status", lambda x: round(x.mean()*100, 2)),
                avg_loan=("loan_amnt",      lambda x: round(x.mean(), 2)),
                avg_rate=("loan_int_rate",  lambda x: round(x.mean(), 2)),
            ).reset_index()
        )
    return (
        _a("loan_grade").sort_values("loan_grade"),
        _a("loan_intent").sort_values("default_rate", ascending=False),
        _a("cb_person_default_on_file"),
        _a("income_group").sort_values("default_rate", ascending=False),
        _a("person_home_ownership").sort_values("default_rate", ascending=False),
        _a("employment_group").sort_values("default_rate", ascending=False),
    )


grade_df, intent_df, hist_df, income_df, home_df, emp_df = agg_all(df)

# ═══════════════════════════════════════════════════════════════════════════
# 1 · OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if nav == "📊  Overview":
    st.markdown("# 📊 Credit Risk Overview")
    st.caption(
        f"Dataset · **{len(raw_df):,} rows** · **{raw_df.shape[1]} columns** "
        f"→ after cleaning: **{len(df):,} rows**")
    st.divider()

    total  = len(df)
    deflt  = int(df["loan_status"].sum())
    safe   = total - deflt
    dr     = deflt / total * 100
    avgl   = df["loan_amnt"].mean()
    avgr   = df["loan_int_rate"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Total Borrowers",   f"{total:,}",      "in dataset",              C_INDIGO)
    with c2: kpi("Defaulted Loans",   f"{deflt:,}",      f"{dr:.1f}% rate",         C_RED)
    with c3: kpi("Safe Borrowers",    f"{safe:,}",       f"{100-dr:.1f}% safe",     C_GREEN)
    with c4: kpi("Avg Loan Amount",   f"${avgl:,.0f}",   "per borrower",            C_AMBER)
    with c5: kpi("Avg Interest Rate", f"{avgr:.2f}%",    "portfolio avg",           C_PURPLE)

    interpretation_box(
        "Portfolio interpretation",
        [
            f"Observed default rate is {dr:.1f}% ({deflt:,} of {total:,} borrowers), so the dataset is meaningfully imbalanced but still contains a sizable risky segment.",
            f"Average loan amount is ${avgl:,.0f} and average interest rate is {avgr:.2f}%, which gives a quick view of the portfolio’s central tendency.",
            "Use the grade, intent, and income visuals together: a single chart should not be treated as causal proof on its own.",
        ],
        tone="info",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns([1, 1.7, 1.7])

    # ── Donut
    with r1:
        sec("Risk Distribution")
        rc = df["risk_segment"].value_counts().reset_index()
        rc.columns = ["Segment", "Count"]
        fig = px.pie(rc, names="Segment", values="Count", hole=0.56,
                     color="Segment",
                     color_discrete_map={"High Risk": C_RED, "Low Risk": C_GREEN})
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont=dict(size=15, color="white"),
            marker=dict(line=dict(color="#FBF8F5", width=3)))
        fig.update_layout(**chart_base(340), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Grade bars
    with r2:
        sec("Default Rate by Loan Grade")
        fig = go.Figure(go.Bar(
            x=grade_df["loan_grade"],
            y=grade_df["default_rate"],
            marker_color=[GRADE_COLORS.get(g, "#888") for g in grade_df["loan_grade"]],
            text=[f"{v}%" for v in grade_df["default_rate"]],
            textposition="outside",
            textfont=dict(size=14, color="#1C1C1C"),
            width=0.55,
        ))
        fig.update_layout(**chart_base(340))
        apply_axes(fig, xtitle="Loan Grade", ytitle="Default Rate (%)",
                   yrange=[0, 115])
        st.plotly_chart(fig, use_container_width=True)

    # ── Home ownership
    with r3:
        sec("Avg Loan Amount — Home Ownership")
        hl = (df.groupby("person_home_ownership")["loan_amnt"]
                .mean().reset_index()
                .sort_values("loan_amnt"))
        fig = go.Figure(go.Bar(
            x=hl["loan_amnt"],
            y=hl["person_home_ownership"],
            orientation="h",
            marker=dict(
                color=hl["loan_amnt"],
                colorscale=[[0, "#F3E5AB"], [1, C_AMBER]],
                showscale=False,
            ),
            text=hl["loan_amnt"].apply(lambda x: f"${x:,.0f}"),
            textposition="outside",
            textfont=dict(size=14, color="#1C1C1C"),
        ))
        fig.update_layout(**chart_base(340))
        apply_axes(fig, xtitle="Avg Loan Amount ($)", ytitle="")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    r4, r5 = st.columns(2)

    with r4:
        sec("Loan Amount Distribution — Risk Segment")
        fig = px.histogram(df, x="loan_amnt", color="risk_segment",
                           nbins=60, barmode="overlay", opacity=0.75,
                           color_discrete_map={"High Risk": C_RED, "Low Risk": C_GREEN})
        fig.update_layout(**chart_base(340))
        fig.update_layout(legend=dict(font=dict(size=17), orientation="h", y=1.08))
        apply_axes(fig, xtitle="Loan Amount ($)", ytitle="Count")
        st.plotly_chart(fig, use_container_width=True)

    with r5:
        sec("Loan Intent Volume")
        ic = df["loan_intent"].value_counts().reset_index()
        ic.columns = ["Intent", "Count"]
        fig = go.Figure(go.Bar(
            x=ic["Intent"],
            y=ic["Count"],
            marker_color=[C_INDIGO, C_TEAL, C_AMBER, C_RED,
                          C_GREEN, C_PURPLE][:len(ic)],
            text=ic["Count"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(size=13, color="#1C1C1C"),
        ))
        fig.update_layout(**chart_base(340))
        apply_axes(fig, xtitle="Loan Intent", ytitle="Count")
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 2 · DEEP ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "🔍  Deep Analytics":
    st.markdown("# 🔍 Deep Analytics")
    st.divider()

    with st.expander("🔧  Filter Data", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            g_f = st.multiselect("Loan Grade",
                sorted(df["loan_grade"].unique()), sorted(df["loan_grade"].unique()))
        with f2:
            i_f = st.multiselect("Loan Intent",
                sorted(df["loan_intent"].unique()), sorted(df["loan_intent"].unique()))
        with f3:
            h_f = st.multiselect("Home Ownership",
                sorted(df["person_home_ownership"].unique()),
                sorted(df["person_home_ownership"].unique()))
        with f4:
            s_f = st.multiselect("Risk Segment",
                ["High Risk", "Low Risk"], ["High Risk", "Low Risk"])
        age_r = st.slider("Age Range",
            int(df["person_age"].min()), int(df["person_age"].max()),
            (int(df["person_age"].min()), int(df["person_age"].max())))

    dff = df[
        df["loan_grade"].isin(g_f) &
        df["loan_intent"].isin(i_f) &
        df["person_home_ownership"].isin(h_f) &
        df["risk_segment"].isin(s_f) &
        df["person_age"].between(*age_r)
    ]
    st.markdown(
        f"<p style='font-size:1.05rem;font-weight:700;color:#333;'>"
        f"✅ &nbsp; {len(dff):,} rows match your filters</p>",
        unsafe_allow_html=True)
    st.divider()

    t1, t2, t3, t4 = st.tabs(
        ["📦  Box Plots", "🎻  Violin Plots", "🔵  Scatter Matrix", "🌞  Sunburst"])

    with t1:
        st.markdown("<br>", unsafe_allow_html=True)
        bc1, bc2 = st.columns(2)
        with bc1:
            feat_b = st.selectbox("Feature", NUMERIC_FEATS,
                                   index=NUMERIC_FEATS.index("loan_int_rate"),
                                   key="bf")
        with bc2:
            grp_b  = st.radio("Group By",
                ["risk_segment", "loan_grade", "income_group", "employment_group"],
                horizontal=True, key="bg")
        fig = px.box(dff, x=grp_b, y=feat_b, color=grp_b,
                     color_discrete_sequence=[C_RED, C_INDIGO, C_AMBER,
                                              C_GREEN, C_TEAL, C_PURPLE],
                     points="outliers")
        fig.update_traces(marker=dict(size=4, opacity=0.45))
        fig.update_layout(**chart_base(440), showlegend=False)
        apply_axes(fig,
                   xtitle=grp_b.replace("_", " ").title(),
                   ytitle=feat_b.replace("_", " ").title())
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.markdown("<br>", unsafe_allow_html=True)
        feat_v = st.selectbox("Feature", NUMERIC_FEATS, key="vf")
        fig = px.violin(dff, y=feat_v, x="risk_segment", color="risk_segment",
                        box=True, points=False,
                        color_discrete_map={"High Risk": C_RED, "Low Risk": C_GREEN})
        fig.update_layout(**chart_base(440), showlegend=False)
        apply_axes(fig, xtitle="Risk Segment", ytitle=feat_v.replace("_"," ").title())
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        st.markdown("<br>", unsafe_allow_html=True)
        dflt_sm = ["loan_int_rate", "loan_percent_income",
                   "person_income", "loan_amnt"]
        sm_f = st.multiselect("Pick 2–5 Features", NUMERIC_FEATS,
                               default=dflt_sm, key="sm")
        if len(sm_f) < 2:
            st.info("Select at least 2 features.")
        else:
            samp = dff.sample(min(3000, len(dff)), random_state=42)
            fig = px.scatter_matrix(
                samp, dimensions=sm_f, color="risk_segment",
                color_discrete_map={"High Risk": C_RED, "Low Risk": C_GREEN},
                opacity=0.45)
            fig.update_traces(diagonal_visible=False,
                              showupperhalf=False,
                              marker=dict(size=4))
            fig.update_layout(**chart_base(600))
            st.plotly_chart(fig, use_container_width=True)

    with t4:
        st.markdown("<br>", unsafe_allow_html=True)
        ss = dff.sample(min(10000, len(dff)), random_state=1)
        fig = px.sunburst(ss,
                          path=["income_group", "loan_grade", "risk_segment"],
                          color="loan_grade",
                          color_discrete_map={**GRADE_COLORS, "(?)": "#CCC"})
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            font=dict(size=14, color="#1C1C1C"),
            height=520,
            margin=dict(t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 3 · AGGREGATIONS
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "📈  Aggregations":
    st.markdown("# 📈 Aggregations & Drill-Downs")
    st.divider()

    drill = st.multiselect(
        "🔎  Drill-down — filter by Loan Grade (all tabs)",
        sorted(df["loan_grade"].unique()), sorted(df["loan_grade"].unique()))
    dff = df[df["loan_grade"].isin(drill)]
    gd, ind, hid, ind2, hmd, epd = agg_all(dff)

    def dual(df_a, xcol):
        """Dual-axis chart — no yaxis conflict."""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        bar_colors = ([GRADE_COLORS.get(g, C_INDIGO) for g in df_a[xcol]]
                      if xcol == "loan_grade" else [C_INDIGO] * len(df_a))
        fig.add_trace(go.Bar(
            x=df_a[xcol], y=df_a["total"],
            name="Total Loans",
            marker_color=bar_colors, opacity=0.82,
            text=df_a["total"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(size=13, color="#1C1C1C"),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df_a[xcol], y=df_a["default_rate"],
            name="Default Rate (%)", mode="lines+markers",
            line=dict(color=C_RED, width=3),
            marker=dict(size=10, color=C_RED,
                        line=dict(color="white", width=2)),
        ), secondary_y=True)
        # Use update_yaxes with secondary_y selector — no dict conflict
        fig.update_yaxes(title_text="Total Loans",
                         title_font=dict(size=14, color="#1C1C1C"),
                         tickfont=dict(size=13),
                         gridcolor="#EDE8E0",
                         secondary_y=False)
        fig.update_yaxes(title_text="Default Rate (%)",
                         title_font=dict(size=14, color=C_RED),
                         tickfont=dict(size=13, color=C_RED),
                         secondary_y=True)
        fig.update_xaxes(tickfont=dict(size=14), title_font=dict(size=14))
        fig.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FAF8F5",
            font=dict(size=14, color="#1C1C1C"),
            legend=dict(font=dict(size=13), orientation="h", y=1.1),
            height=400,
        )
        return fig

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Loan Grade", "Loan Intent", "Default History",
        "Income Group", "Home Ownership", "Employment"])

    with t1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(dual(gd, "loan_grade"), use_container_width=True)
        st.dataframe(gd, hide_index=True, use_container_width=True)

    with t2:
        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=ind["default_rate"], y=ind["loan_intent"],
            orientation="h",
            marker=dict(
                color=ind["default_rate"],
                colorscale=[[0,"#C8E6C9"],[0.5,"#FFF9C4"],[1,"#FFCDD2"]],
                showscale=False,
            ),
            text=ind["default_rate"].apply(lambda x: f"{x}%"),
            textposition="outside",
            textfont=dict(size=14, color="#1C1C1C"),
        ))
        fig.update_layout(**chart_base(380))
        apply_axes(fig, xtitle="Default Rate (%)", ytitle="")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ind, hide_index=True, use_container_width=True)

    with t3:
        st.markdown("<br>", unsafe_allow_html=True)
        hid["label"] = hid["cb_person_default_on_file"].map(
            {"Y": "Prior Default  (Y)", "N": "No Prior Default  (N)"})
        fig = go.Figure(go.Bar(
            x=hid["label"], y=hid["default_rate"],
            marker_color=[C_RED, C_GREEN][:len(hid)],
            text=hid["default_rate"].apply(lambda x: f"{x}%"),
            textposition="outside",
            textfont=dict(size=16, color="#1C1C1C"),
            width=0.45,
        ))
        fig.update_layout(**chart_base(380), showlegend=False)
        apply_axes(fig, xtitle="Prior Default History",
                   ytitle="Default Rate (%)", yrange=[0, 50])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(hid, hide_index=True, use_container_width=True)

    with t4:
        st.markdown("<br>", unsafe_allow_html=True)
        ig_colors = {"Low Income": C_RED,
                     "Middle Income": C_AMBER,
                     "High Income": C_GREEN}
        fig = go.Figure([
            go.Bar(
                x=[row["income_group"]], y=[row["default_rate"]],
                name=str(row["income_group"]),
                marker_color=ig_colors.get(str(row["income_group"]), C_INDIGO),
                text=[f"{row['default_rate']}%"],
                textposition="outside",
                textfont=dict(size=16, color="#1C1C1C"),
                width=0.45,
            )
            for _, row in ind2.iterrows()
        ])
        fig.update_layout(**chart_base(380), showlegend=False)
        apply_axes(fig, xtitle="Income Group",
                   ytitle="Default Rate (%)", yrange=[0, 60])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ind2, hide_index=True, use_container_width=True)

    with t5:
        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=hmd["person_home_ownership"], y=hmd["default_rate"],
            marker=dict(
                color=hmd["default_rate"],
                colorscale=[[0,"#C8E6C9"],[1,"#FFCDD2"]],
                showscale=True,
                colorbar=dict(title="Default %",
                              tickfont=dict(size=13)),
            ),
            text=hmd["default_rate"].apply(lambda x: f"{x}%"),
            textposition="outside",
            textfont=dict(size=14, color="#1C1C1C"),
        ))
        fig.update_layout(**chart_base(380))
        apply_axes(fig, xtitle="Home Ownership", ytitle="Default Rate (%)",
                   yrange=[0, 42])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(hmd, hide_index=True, use_container_width=True)

    with t6:
        st.markdown("<br>", unsafe_allow_html=True)
        ep_c = [C_RED, C_AMBER, C_GREEN]
        fig = go.Figure([
            go.Bar(
                x=[row["employment_group"]], y=[row["default_rate"]],
                name=str(row["employment_group"]),
                marker_color=ep_c[i % len(ep_c)],
                text=[f"{row['default_rate']}%"],
                textposition="outside",
                textfont=dict(size=15, color="#1C1C1C"),
                width=0.45,
            )
            for i, (_, row) in enumerate(epd.iterrows())
        ])
        fig.update_layout(**chart_base(380), showlegend=False)
        apply_axes(fig, xtitle="Employment Group",
                   ytitle="Default Rate (%)", yrange=[0, 40])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(epd, hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 4 · CORRELATION & HEATMAP
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "🔗  Correlation & Heatmap":
    st.markdown("# 🔗 Correlation & Heatmap")
    st.caption("Pearson correlation between numeric features and loan default status.")
    st.divider()

    num_df = df[NUMERIC_FEATS + ["loan_status"]].copy()
    cm = num_df.corr()
    cols = cm.columns.tolist()
    z    = cm.values.round(3)

    sec("Full Numeric Correlation Heatmap")
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=cols,
        colorscale=[[0,"#2E7D32"],[0.5,"#FFFFFF"],[1,"#D32F2F"]],
        zmid=0, zmin=-1, zmax=1,
        text=z, texttemplate="<b>%{text:.2f}</b>",
        textfont=dict(size=13),
        hovertemplate="%{x} × %{y}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        font=dict(size=14, color="#1C1C1C"),
        height=520,
        margin=dict(t=20, b=60, l=20, r=20),
    )
    fig.update_xaxes(tickfont=dict(size=12), tickangle=-30)
    fig.update_yaxes(tickfont=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    sec("Feature Correlation with loan_status  (target variable)")
    corr_ls = (
        cm["loan_status"].drop("loan_status")
        .reset_index()
        .rename(columns={"index": "Feature", "loan_status": "Correlation"})
        .sort_values("Correlation")
    )
    bar_cols = [C_GREEN if v < 0 else C_RED for v in corr_ls["Correlation"]]
    fig = go.Figure(go.Bar(
        x=corr_ls["Correlation"], y=corr_ls["Feature"],
        orientation="h",
        marker_color=bar_cols,
        text=corr_ls["Correlation"].apply(lambda x: f"{x:+.4f}"),
        textposition="outside",
        textfont=dict(size=14, color="#1C1C1C"),
        width=0.6,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#999", line_width=2)
    fig.update_layout(**chart_base(400))
    apply_axes(fig, xtitle="Pearson r  with  loan_status", ytitle="")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    sec("Top Risk Driver Scatters")
    sc1, sc2 = st.columns(2)
    samp = df.sample(min(5000, len(df)), random_state=42)

    with sc1:
        fig = px.scatter(samp, x="loan_percent_income", y="loan_int_rate",
                         color="risk_segment", opacity=0.45,
                         color_discrete_map={"High Risk": C_RED,
                                             "Low Risk": C_GREEN})
        fig.update_layout(**chart_base(380),
                          title=dict(text="Loan % Income  vs  Interest Rate",
                                     font=dict(size=18, color="#1C1C1C")))
        fig.update_layout(legend=dict(font=dict(size=16)))
        apply_axes(fig, xtitle="Loan % of Income",
                   ytitle="Interest Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with sc2:
        fig = px.scatter(samp, x="person_income", y="loan_amnt",
                         color="risk_segment", opacity=0.45,
                         color_discrete_map={"High Risk": C_RED,
                                             "Low Risk": C_GREEN})
        fig.update_layout(**chart_base(380),
                          title=dict(text="Annual Income  vs  Loan Amount",
                                     font=dict(size=18, color="#1C1C1C")))
        fig.update_layout(legend=dict(font=dict(size=16)))
        apply_axes(fig, xtitle="Annual Income ($)",
                   ytitle="Loan Amount ($)")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 5 · ML MODELS
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "🤖  ML Models":
    st.markdown("# 🤖 ML Model Performance")
    st.caption("Random Forest vs Logistic Regression — trained on 80 %, evaluated on 20 %.")
    st.divider()

    rf_acc = M["rf_rep"]["accuracy"]
    lr_acc = M["lr_rep"]["accuracy"]
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Random Forest AUC",  str(M["rf_auc"]), "Area under ROC",  C_INDIGO)
    with c2: kpi("RF Accuracy",        f"{rf_acc:.3f}",   "on 20% test set", C_GREEN)
    with c3: kpi("Logistic Reg. AUC",  str(M["lr_auc"]), "Area under ROC",  C_AMBER)
    with c4: kpi("LR Accuracy",        f"{lr_acc:.3f}",   "on 20% test set", C_TEAL)

    st.divider()
    better_model = "Random Forest" if M["rf_auc"] >= M["lr_auc"] else "Logistic Regression"
    interpretation_box(
        "Model interpretation",
        [
            f"Random Forest AUC = {M['rf_auc']:.4f} and Logistic Regression AUC = {M['lr_auc']:.4f}; on ranking quality, {better_model} is stronger on this test split.",
            f"RF accuracy = {rf_acc:.3f} and LR accuracy = {lr_acc:.3f}; accuracy should be read together with ROC and precision-recall, not in isolation.",
            "These metrics describe performance on the held-out test set from this dataset. They do not guarantee perfect predictions for future applicants.",
        ],
        tone="warn" if abs(M['rf_auc']-M['lr_auc']) < 0.03 else "good",
    )

    t1, t2, t3, t4 = st.tabs([
        "📉  ROC Curves", "🟦  Confusion Matrix",
        "🌟  Feature Importance", "📐  Precision-Recall"])

    with t1:
        st.markdown("<br>", unsafe_allow_html=True)
        rf_fpr, rf_tpr, _ = roc_curve(M["y_te"], M["rf_p"])
        lr_fpr, lr_tpr, _ = roc_curve(M["y_te"], M["lr_p"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, mode="lines",
                                  name=f"Random Forest  (AUC = {M['rf_auc']})",
                                  line=dict(color=C_INDIGO, width=3)))
        fig.add_trace(go.Scatter(x=lr_fpr, y=lr_tpr, mode="lines",
                                  name=f"Logistic Regression  (AUC = {M['lr_auc']})",
                                  line=dict(color=C_AMBER, width=3)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  line=dict(dash="dash", color="#AAA", width=2),
                                  showlegend=False))
        fig.update_layout(**chart_base(440))
        fig.update_layout(legend=dict(font=dict(size=16), x=0.42, y=0.05))
        apply_axes(fig, xtitle="False Positive Rate",
                   ytitle="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.markdown("<br>", unsafe_allow_html=True)
        ms = st.radio("Model", ["Random Forest", "Logistic Regression"],
                      horizontal=True, key="cm_m")
        preds = M["rf_pred"] if ms == "Random Forest" else M["lr_pred"]
        clr   = [[0,"#EEF2FF"],[1,C_INDIGO]] \
            if ms == "Random Forest" else [[0,"#FFF7ED"],[1,C_AMBER]]
        cm_mat = confusion_matrix(M["y_te"], preds)
        labels = ["Non-Default (0)", "Default (1)"]
        fig = ff.create_annotated_heatmap(
            cm_mat, x=labels, y=labels,
            colorscale=clr, showscale=True,
            font_colors=["#1C1C1C", "#1C1C1C"],
        )
        for ann in fig.layout.annotations:
            ann.font = dict(size=20, color="#1C1C1C")
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            font=dict(size=15, color="#1C1C1C"),
            height=400,
            margin=dict(t=40, b=50),
        )
        fig.update_xaxes(title_text="Predicted",
                         title_font=dict(size=15), tickfont=dict(size=14))
        fig.update_yaxes(title_text="Actual",
                         title_font=dict(size=15), tickfont=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)

        rep = M["rf_rep"] if ms == "Random Forest" else M["lr_rep"]
        rep_df = pd.DataFrame({
            "Class":     ["Non-Default (0)", "Default (1)", "Macro Avg"],
            "Precision": [rep["0"]["precision"], rep["1"]["precision"],
                          rep["macro avg"]["precision"]],
            "Recall":    [rep["0"]["recall"],    rep["1"]["recall"],
                          rep["macro avg"]["recall"]],
            "F1-Score":  [rep["0"]["f1-score"],  rep["1"]["f1-score"],
                          rep["macro avg"]["f1-score"]],
        }).round(4)
        st.dataframe(rep_df, hide_index=True, use_container_width=True)

    with t3:
        st.markdown("<br>", unsafe_allow_html=True)
        flabels = [f.replace("_enc","").replace("_"," ").title()
                   for f in FEAT_COLS]
        imp_df = (pd.DataFrame({"Feature": flabels,
                                 "Importance": M["feat_imp"]})
                    .sort_values("Importance"))
        fig = go.Figure(go.Bar(
            x=imp_df["Importance"],
            y=imp_df["Feature"],
            orientation="h",
            marker=dict(
                color=imp_df["Importance"],
                colorscale=[[0,"#E8EAF6"],[1,C_INDIGO]],
                showscale=False,
            ),
            text=imp_df["Importance"].apply(lambda x: f"{x:.4f}"),
            textposition="outside",
            textfont=dict(size=14, color="#1C1C1C"),
        ))
        fig.update_layout(**chart_base(470))
        apply_axes(fig, xtitle="Feature Importance", ytitle="")
        st.plotly_chart(fig, use_container_width=True)

    with t4:
        st.markdown("<br>", unsafe_allow_html=True)
        rf_p2, rf_r2, _ = precision_recall_curve(M["y_te"], M["rf_p"])
        lr_p2, lr_r2, _ = precision_recall_curve(M["y_te"], M["lr_p"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rf_r2, y=rf_p2, mode="lines",
                                  name="Random Forest",
                                  line=dict(color=C_INDIGO, width=3)))
        fig.add_trace(go.Scatter(x=lr_r2, y=lr_p2, mode="lines",
                                  name="Logistic Regression",
                                  line=dict(color=C_AMBER, width=3)))
        fig.update_layout(**chart_base(420))
        fig.update_layout(legend=dict(font=dict(size=16)))
        apply_axes(fig, xtitle="Recall", ytitle="Precision")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 6 · LOAN RISK PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "🎯  Loan Risk Predictor":
    st.markdown("# 🎯 Loan Risk Predictor")
    st.caption("Fill in the applicant details. Both models will score the applicant "
               "and the result is saved automatically to the Footprint Database.")
    st.divider()

    with st.form("pred_form"):
        st.markdown("""
        <div style="background:#EEF2FF;border-radius:12px;padding:1rem 1.5rem;
                    border-left:5px solid #283593;margin-bottom:1.4rem;">
            <p style="font-size:1.05rem;font-weight:800;color:#1A2340;margin:0;">
                🏢 &nbsp; Institution &amp; Applicant Details
            </p>
        </div>""", unsafe_allow_html=True)

        tc1, tc2 = st.columns([1, 3])
        with tc1:
            institution = st.text_input("Institution Name", value="Bank A")

        st.markdown("<br>", unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns(3)

        with pc1:
            st.markdown("**👤 Personal Info**")
            p_age  = st.number_input("Age (years)",     18, 100, 30)
            p_inc  = st.number_input("Annual Income ($)", 1000, 5_000_000,
                                      55000, step=1000)
            p_home = st.selectbox("Home Ownership",
                                   LABEL_MAPS["person_home_ownership"])
            p_emp  = st.number_input("Employment Length (yrs)",
                                      0.0, 50.0, 4.0, step=0.5)

        with pc2:
            st.markdown("**📋 Loan Details**")
            l_int  = st.selectbox("Loan Intent",  LABEL_MAPS["loan_intent"])
            l_grd  = st.selectbox("Loan Grade",   LABEL_MAPS["loan_grade"])
            l_amt  = st.number_input("Loan Amount ($)",
                                      500, 500_000, 10000, step=500)
            l_rate = st.number_input("Interest Rate (%)", 1.0, 35.0, 11.0, step=0.1)

        with pc3:
            st.markdown("**🏦 Credit Profile**")
            l_pct  = st.number_input("Loan % of Income", 0.01, 1.0, 0.18, step=0.01)
            l_dfl  = st.selectbox("Prior Default on File",
                                   LABEL_MAPS["cb_person_default_on_file"],
                                   format_func=lambda x: "Yes (Y)" if x=="Y" else "No (N)")
            l_crd  = st.number_input("Credit History (yrs)", 1, 50, 5)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🚀  Predict Default Risk", use_container_width=True)

    if submitted:
        applicant = {
            "person_age": p_age, "person_income": p_inc,
            "person_home_ownership": p_home, "person_emp_length": p_emp,
            "loan_intent": l_int, "loan_grade": l_grd,
            "loan_amnt": l_amt, "loan_int_rate": l_rate,
            "loan_percent_income": l_pct,
            "cb_person_default_on_file": l_dfl,
            "cb_person_cred_hist_length": l_crd,
        }
        X_in    = encode_for_model(applicant)
        X_in_sc = M["sc"].transform(X_in)
        rf_prob = float(M["rf"].predict_proba(X_in)[0, 1])
        lr_prob = float(M["lr"].predict_proba(X_in_sc)[0, 1])
        avg_pr  = (rf_prob + lr_prob) / 2
        pred    = 1 if avg_pr >= 0.5 else 0

        st.divider()
        if pred == 1:
            st.markdown(f"""
            <div style="background:#FFF5F5;border:2.5px solid {C_RED};
                        border-radius:16px;padding:2.2rem 2rem;text-align:center;">
                <div style="font-size:3.2rem;">⚠️</div>
                <div style="font-size:2rem;font-weight:900;color:{C_RED};
                            margin:0.5rem 0;letter-spacing:-0.01em;">
                    HIGH DEFAULT RISK
                </div>
                <div style="font-size:1.2rem;color:#7B1212;font-weight:600;">
                    Ensemble probability: &nbsp;<strong>{avg_pr*100:.1f}%</strong>
                </div>
                <div style="font-size:1rem;color:#9B2020;margin-top:0.5rem;">
                    Applicant is predicted to <strong>DEFAULT</strong> on this loan.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#F0FBF0;border:2.5px solid {C_GREEN};
                        border-radius:16px;padding:2.2rem 2rem;text-align:center;">
                <div style="font-size:3.2rem;">✅</div>
                <div style="font-size:2rem;font-weight:900;color:{C_GREEN};
                            margin:0.5rem 0;letter-spacing:-0.01em;">
                    LOW DEFAULT RISK
                </div>
                <div style="font-size:1.2rem;color:#1B5E20;font-weight:600;">
                    Ensemble probability: &nbsp;<strong>{avg_pr*100:.1f}%</strong>
                </div>
                <div style="font-size:1rem;color:#276127;margin-top:0.5rem;">
                    Applicant is predicted to <strong>REPAY</strong> the loan.
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        kc1, kc2, kc3 = st.columns(3)
        with kc1: kpi("Random Forest",       f"{rf_prob*100:.1f}%",
                        "Default probability", C_INDIGO)
        with kc2: kpi("Logistic Regression", f"{lr_prob*100:.1f}%",
                        "Default probability", C_AMBER)
        with kc3: kpi("Ensemble Average",    f"{avg_pr*100:.1f}%",
                        "Final score",
                        C_RED if pred else C_GREEN)

        interpretation_box(
            "Prediction interpretation",
            [
                f"Random Forest estimates default probability at {rf_prob*100:.1f}% and Logistic Regression at {lr_prob*100:.1f}%.",
                f"The dashboard uses the simple average of both models ({avg_pr*100:.1f}%) as the final decision score.",
                "This output is a model-based risk signal for screening support, not a guarantee that the borrower will definitely default or repay.",
            ],
            tone="warn" if pred == 1 else "good",
        )

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_pr * 100,
            number={"suffix": "%",
                    "font": {"size": 52, "color": "#1C1C1C",
                             "family": "Segoe UI"}},
            title={"text": "<b>Default Probability</b>",
                   "font": {"size": 20, "color": "#1C1C1C"}},
            delta={"reference": 50,
                   "increasing": {"color": C_RED},
                   "decreasing": {"color": C_GREEN},
                   "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100],
                         "tickwidth": 2,
                         "tickcolor": "#555",
                         "tickfont": {"size": 16, "color": "#333"}},
                "bar":  {"color": C_RED if avg_pr >= 0.5 else C_GREEN,
                         "thickness": 0.28},
                "bgcolor": "#FBF8F5",
                "borderwidth": 2,
                "bordercolor": "#CCC",
                "steps": [
                    {"range": [0,  30], "color": "#C8E6C9"},
                    {"range": [30, 60], "color": "#FFF9C4"},
                    {"range": [60,100], "color": "#FFCDD2"},
                ],
                "threshold": {
                    "line": {"color": "#333", "width": 3},
                    "thickness": 0.8, "value": 50},
            },
        ))
        fig.update_layout(
            height=360, paper_bgcolor="#FFFFFF",
            font=dict(family="Segoe UI, Arial", color="#1C1C1C"),
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Save
        append_log({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "institution": institution,
            **applicant,
            "rf_probability_pct":  round(rf_prob * 100, 2),
            "lr_probability_pct":  round(lr_prob * 100, 2),
            "predicted_default":   pred,
        })
        st.success("✅  Prediction saved — navigate to **🗄️ Footprint Database** to view all records.")

# ═══════════════════════════════════════════════════════════════════════════
# 7 · FOOTPRINT DATABASE
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "🗄️  Footprint Database":
    st.markdown("# 🗄️ Footprint Database")
    st.caption("Every applicant scored through the Predictor is stored here permanently — "
               "growing historical default footprints for future model retraining.")
    st.divider()

    log_df = load_log()

    if log_df.empty:
        st.markdown("""
        <div style="background:#FFFFFF;border-radius:16px;padding:3.5rem;
                    text-align:center;box-shadow:0 2px 16px rgba(0,0,0,0.08);
                    border-top:5px solid #283593;">
            <div style="font-size:4rem;">🗄️</div>
            <h2 style="color:#0F1628;margin:1rem 0 0.6rem;">No Records Yet</h2>
            <p style="font-size:1.1rem;color:#666;max-width:440px;margin:0 auto;">
                Go to <strong>🎯 Loan Risk Predictor</strong> and assess your
                first applicant. Records appear here automatically.
            </p>
        </div>""", unsafe_allow_html=True)
        st.stop()

    tot  = len(log_df)
    hi   = int((log_df["predicted_default"]==1).sum()) \
           if "predicted_default" in log_df.columns else 0
    lo   = tot - hi
    rate = hi / tot * 100 if tot else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("Total Assessed", f"{tot:,}",   "all time",          C_INDIGO)
    with k2: kpi("High Risk",      f"{hi:,}",    "predicted default", C_RED)
    with k3: kpi("Low Risk",       f"{lo:,}",    "predicted safe",    C_GREEN)
    with k4: kpi("Risk Rate",      f"{rate:.1f}%","of assessed",      C_AMBER)

    interpretation_box(
        "Database interpretation",
        [
            f"The Footprint Database currently stores {tot:,} assessed applicants, of which {hi:,} are flagged high risk.",
            f"The saved high-risk share is {rate:.1f}%. This reflects model output on assessed cases, not the original raw dataset default rate.",
            "Use this section to monitor how your own assessed pipeline evolves over time and across institutions.",
        ],
        tone="info",
    )

    st.divider()

    if tot >= 2:
        fa1, fa2 = st.columns(2)
        with fa1:
            sec("Risk Split of Assessed Applicants")
            rc = (log_df["predicted_default"]
                  .map({1:"High Risk",0:"Low Risk"})
                  .value_counts().reset_index())
            rc.columns = ["Segment","Count"]
            fig = px.pie(rc, names="Segment", values="Count", hole=0.5,
                         color="Segment",
                         color_discrete_map={"High Risk":C_RED,"Low Risk":C_GREEN})
            fig.update_traces(textfont=dict(size=15),
                              marker=dict(line=dict(color="#FBF8F5",width=3)))
            fig.update_layout(**chart_base(320))
            st.plotly_chart(fig, use_container_width=True)

        with fa2:
            sec("Assessments Over Time")
            if "timestamp" in log_df.columns:
                log_df["date"] = pd.to_datetime(log_df["timestamp"]).dt.date
                tl = log_df.groupby("date").size().reset_index(name="Assessments")
                fig = px.area(tl, x="date", y="Assessments",
                               color_discrete_sequence=[C_INDIGO], markers=True)
                fig.update_layout(**chart_base(320))
                apply_axes(fig, xtitle="Date", ytitle="Assessments")
                st.plotly_chart(fig, use_container_width=True)

        if "rf_probability_pct" in log_df.columns and tot >= 3:
            sec("RF Probability Distribution")
            fig = px.histogram(log_df, x="rf_probability_pct",
                               color="predicted_default",
                               color_discrete_map={1:C_RED,0:C_GREEN},
                               nbins=20, barmode="overlay", opacity=0.82)
            fig.update_layout(**chart_base(320))
            apply_axes(fig, xtitle="RF Default Probability (%)",
                       ytitle="Count")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    sec("All Assessment Records")

    view = log_df.copy()
    vf1, vf2 = st.columns([2,1])
    with vf1:
        if "institution" in log_df.columns:
            inst_opts = sorted(log_df["institution"].dropna().unique())
            inst_f = st.multiselect("Filter by Institution",
                                     inst_opts, default=inst_opts)
            view = view[view["institution"].isin(inst_f)]
    with vf2:
        if "predicted_default" in log_df.columns:
            rf_filter = st.multiselect(
                "Risk Level", [0,1], default=[0,1],
                format_func=lambda x: "Low Risk (0)" if x==0 else "High Risk (1)")
            view = view[view["predicted_default"].isin(rf_filter)]

    st.dataframe(view.reset_index(drop=True),
                 use_container_width=True, height=420)

    st.download_button(
        "⬇️  Export as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="credlens_footprint.csv",
        mime="text/csv",
    )
    st.info(
        "💡 **Production tip:** Replace `prediction_log.csv` with PostgreSQL "
        "or SQLite for concurrent multi-institution writes and long-term scalability."
    )

# ═══════════════════════════════════════════════════════════════════════════
# 8 · RISK CHATBOT
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "💬  Risk Chatbot":
    st.markdown("# 💬 Risk Chatbot")
    st.caption("Ask questions about the dataset, model performance, loan grades, loan intents, and prediction records. The assistant uses the current dashboard data and gives grounded answers only.")
    st.divider()

    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.markdown("**Try asking:**\n- What is the overall default rate?\n- Which loan grade is riskiest?")
    with ex2:
        st.markdown("**You can also ask:**\n- Which loan intent has the highest default rate?\n- What are the top model drivers?")
    with ex3:
        st.markdown("**Operational questions:**\n- How many records are in the Footprint Database?\n- How can I reduce default risk?")

    interpretation_box(
        "How to use the chatbot well",
        [
            "Ask analytical questions, not generic ones. The chatbot is strongest on grades, intents, model metrics, and saved prediction records.",
            "Its answers are grounded in the current uploaded dataset and dashboard outputs, so they remain explainable.",
            "It does not browse the internet and it does not guarantee loan approval decisions.",
        ],
        tone="info",
    )

    if st.button("Use a starter question: What is the overall default rate?", use_container_width=True):
        st.session_state["starter_question"] = "What is the overall default rate?"
    if st.button("Use a starter question: Which loan grade is riskiest?", use_container_width=True):
        st.session_state["starter_question"] = "Which loan grade is riskiest?"
    if st.button("Use a starter question: What are the top model drivers?", use_container_width=True):
        st.session_state["starter_question"] = "What are the top model drivers?"

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Hello. I am the CredLens risk chatbot. I can summarize the portfolio, explain model metrics, "
                    "compare loan grades and intents, and review saved prediction records."
                ),
            }
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input(st.session_state.pop("starter_question", "Ask a question about credit risk analytics..."))
    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        log_df = load_log()
        reply = chatbot_response(user_prompt, df, M, log_df)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    st.divider()
    st.info(
        "This chatbot is analytics-aware, not a generative approval engine. It summarizes the dataset and model outputs already inside the dashboard, so its answers stay explainable and auditable."
    )
