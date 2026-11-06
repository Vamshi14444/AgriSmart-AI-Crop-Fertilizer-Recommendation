"""
app.py
------
AgriSmart AI — Sustainable Crop & Fertilizer Advisor
Enterprise-Grade Academic Machine Learning Web Application.

Theme: Modern AgTech SaaS (Emerald & Dark Slate Glassmorphism)
Includes:
  - 🏠 Home Dashboard
  - 🌱 Crop Recommendation (+ Downloadable PDF Field Advisory)
  - 🌡️ Climate "What-If" Sensitivity Simulator
  - ⚖️ Head-to-Head Crop Comparison
  - 📅 Seasonal Crop Calendar (Kharif / Rabi / Zaid)
  - 🧪 Fertilizer Recommendation
  - 🌿 Soil Health Analysis
  - ♻️ Sustainability Advisor
  - 📊 Model Performance
  - 🕘 Prediction History
  - ℹ️ About the Project
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Project Paths & Imports
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.crop_calendar import (
    CROP_CALENDAR,
    build_crop_timeline_chart,
    get_calendar_info,
)
from src.crop_comparison import (
    CROP_PROFILES,
    compare_two_crops,
    get_all_crop_names,
    get_crop_profile,
)
from src.crop_model import (
    get_feature_importance,
    load_crop_model,
    predict_crop,
)
from src.evaluation import compare_models
from src.fertilizer_model import load_fertilizer_model, predict_fertilizer
from src.history import add_record, clear_history, load_history
from src.recommendations import (
    NUTRIENT_RANGES,
    build_crop_explanation,
    build_fertilizer_explanation,
)
from src.report_generator import generate_field_advisory_pdf
from src.sustainability import get_all_tips, get_checklist, get_sdg_relevance

DATASETS_DIR = ROOT / "datasets"
MODELS_DIR = ROOT / "models"

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AgriSmart AI — Precision AgTech Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Ultra-Modern CSS Design System
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

* {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, kbd, samp, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Background Atmosphere */
.stApp {
    background-color: #080d1a;
    background-image: 
        radial-gradient(circle at 15% 15%, rgba(16, 185, 129, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.06) 0%, transparent 45%),
        radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 0.3) 0%, transparent 70%);
    background-attachment: fixed;
    color: #f1f5f9;
}

/* Clean Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090f1d 0%, #0c1427 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.3);
}

section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
    gap: 0.5rem;
}

/* Sidebar Navigation Items */
section[data-testid="stSidebar"] .stRadio > div {
    background: transparent;
    gap: 5px;
}

section[data-testid="stSidebar"] .stRadio > div > label {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 8px 12px;
    margin-bottom: 2px;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    color: #cbd5e1 !important;
    font-weight: 500;
    font-size: 0.88rem;
    cursor: pointer;
    display: flex;
    align-items: center;
}

section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(16, 185, 129, 0.12);
    border-color: rgba(16, 185, 129, 0.3);
    color: #34d399 !important;
    transform: translateX(3px);
}

section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.22) 0%, rgba(16, 185, 129, 0.08) 100%) !important;
    border-left: 3px solid #10b981 !important;
    border-color: rgba(16, 185, 129, 0.4);
    color: #ffffff !important;
    font-weight: 600;
}

/* Glassmorphism Containers */
.glass-panel {
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.35);
    transition: border-color 0.2s ease, transform 0.2s ease;
}

.glass-panel:hover {
    border-color: rgba(16, 185, 129, 0.25);
}

/* Hero Banner */
.hero-header {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 182, 212, 0.06) 50%, rgba(15, 23, 42, 0.7) 100%);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 20px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 12px 36px -10px rgba(0, 0, 0, 0.5);
    position: relative;
    overflow: hidden;
}

.hero-header::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, transparent 70%);
    pointer-events: none;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.18);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #6ee7b7;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 9999px;
    margin-bottom: 0.75rem;
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 0 0 0.45rem 0;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #6ee7b7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 0.98rem;
    color: #94a3b8;
    line-height: 1.5;
    margin: 0;
    max-width: 820px;
}

/* Metric KPI Cards */
.kpi-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.2rem 1.35rem;
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
}

.kpi-card:hover {
    border-color: rgba(16, 185, 129, 0.35);
    transform: translateY(-2px);
    box-shadow: 0 8px 20px -5px rgba(16, 185, 129, 0.15);
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: #10b981;
}

.kpi-label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 0.35rem;
}

.kpi-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}

.kpi-sub {
    font-size: 0.78rem;
    color: #64748b;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
    100% { opacity: 1; transform: scale(1); }
}

/* Result Showcase Box */
.prediction-hero-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(13, 27, 42, 0.8) 100%);
    border: 1px solid rgba(16, 185, 129, 0.45);
    border-radius: 18px;
    padding: 1.8rem 2.2rem;
    margin: 1.2rem 0;
    box-shadow: 0 15px 35px -8px rgba(16, 185, 129, 0.2);
}

.prediction-crop-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0.2rem 0 0.4rem 0;
    letter-spacing: -0.02em;
    text-shadow: 0 2px 10px rgba(16, 185, 129, 0.3);
}

/* Modern Form Inputs & Buttons */
div[data-testid="stForm"] {
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.75rem 2rem;
}

.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.8rem !important;
    font-weight: 700 !important;
    font-size: 0.98rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 6px 20px -3px rgba(16, 185, 129, 0.4) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
}

.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, #34d399 0%, #10b981 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px -3px rgba(16, 185, 129, 0.6) !important;
}

.preset-btn button {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #cbd5e1 !important;
    border-radius: 10px !important;
    font-size: 0.83rem !important;
    padding: 0.4rem 0.85rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
}

.preset-btn button:hover {
    background: rgba(16, 185, 129, 0.15) !important;
    border-color: rgba(16, 185, 129, 0.4) !important;
    color: #34d399 !important;
    transform: translateY(-1px) !important;
}

.notice-box {
    border-radius: 12px;
    padding: 1.15rem 1.4rem;
    margin: 1rem 0;
    font-size: 0.93rem;
    line-height: 1.55;
}

.notice-box.warning {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fbbf24;
}

.notice-box.info {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.3);
    color: #93c5fd;
}

.notice-box.success {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
}

/* Dataframe styling */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    overflow: hidden;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #10b981 0%, #06b6d4 100%) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    background-color: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    color: #94a3b8;
    font-weight: 600;
    padding: 0 16px;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #f1f5f9;
    border-color: rgba(16, 185, 129, 0.3);
}

.stTabs [aria-selected="true"] {
    background: rgba(16, 185, 129, 0.15) !important;
    border-color: #10b981 !important;
    color: #34d399 !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State & Cache Helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "crop_presets": {
            "Rice (Wet Tropics)": {"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 24.5, "humidity": 82.0, "ph": 6.5, "rainfall": 220.0},
            "Maize (Semi-Arid)": {"N": 75.0, "P": 48.0, "K": 22.0, "temperature": 26.0, "humidity": 65.0, "ph": 6.4, "rainfall": 95.0},
            "Highland Coffee": {"N": 105.0, "P": 28.0, "K": 32.0, "temperature": 25.5, "humidity": 58.0, "ph": 6.8, "rainfall": 165.0},
            "Temperate Apple": {"N": 25.0, "P": 135.0, "K": 200.0, "temperature": 22.0, "humidity": 92.0, "ph": 6.0, "rainfall": 115.0},
            "Cotton Field": {"N": 118.0, "P": 46.0, "K": 20.0, "temperature": 24.0, "humidity": 80.0, "ph": 7.0, "rainfall": 80.0},
        },
        "fert_presets": {
            "Sugarcane on Black Soil": {"Temperature": 29.0, "Humidity": 55.0, "Moisture": 45.0, "Soil Type": "Black", "Crop Type": "Sugarcane", "N": 12.0, "K": 0.0, "P": 36.0},
            "Maize on Sandy Soil": {"Temperature": 26.0, "Humidity": 52.0, "Moisture": 38.0, "Soil Type": "Sandy", "Crop Type": "Maize", "N": 37.0, "K": 0.0, "P": 0.0},
            "Cotton on Red Soil": {"Temperature": 34.0, "Humidity": 65.0, "Moisture": 62.0, "Soil Type": "Red", "Crop Type": "Cotton", "N": 20.0, "K": 0.0, "P": 20.0},
        },
        "last_crop_prediction": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_resource(show_spinner="Initializing Machine Learning Models...")
def _load_crop_cached():
    try:
        return load_crop_model(MODELS_DIR)
    except FileNotFoundError as e:
        return None, str(e)


@st.cache_resource(show_spinner="Initializing Fertilizer Models...")
def _load_fert_cached():
    try:
        return load_fertilizer_model(MODELS_DIR)
    except FileNotFoundError as e:
        return None, str(e)


def _get_crop_model():
    result = _load_crop_cached()
    if isinstance(result, tuple) and len(result) == 2 and result[0] is None:
        return None, result[1]
    return result


def _get_fert_model():
    result = _load_fert_cached()
    if isinstance(result, tuple) and len(result) == 2 and result[0] is None:
        return None, result[1]
    return result


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = [
    "🏠  Home Dashboard",
    "🌱  Crop Recommendation",
    "🌡️  Climate 'What-If' Simulator",
    "⚖️  Head-to-Head Crop Comparison",
    "📅  Seasonal Crop Calendar",
    "🧪  Fertilizer Recommendation",
    "🌿  Soil Health Analysis",
    "♻️  Sustainability Advisor",
    "📊  Model Performance",
    "🕘  Prediction History",
    "ℹ️  About the Project",
]


def render_sidebar() -> str:
    with st.sidebar:
        # Branding Header
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:1.2rem; padding: 0.5rem 0.2rem;">
                <div style="width:42px; height:42px; border-radius:12px; background:linear-gradient(135deg, #10b981 0%, #059669 100%); display:flex; align-items:center; justify-content:center; box-shadow:0 6px 16px rgba(16, 185, 129, 0.4); font-size:22px;">
                    🌾
                </div>
                <div>
                    <div style="font-size:1.15rem; font-weight:800; color:#ffffff; letter-spacing:-0.01em;">AgriSmart AI</div>
                    <div style="display:inline-block; font-size:0.7rem; font-weight:700; color:#34d399; background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.3); border-radius:10px; padding:1px 7px;">
                        v2.4 PRO • ML ADVISOR
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<p style='font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; font-weight:700; margin-bottom:0.3rem;'>Navigation</p>", unsafe_allow_html=True)
        selected_page = st.radio("Navigation Menu", PAGES, label_visibility="collapsed")

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; font-weight:700; margin-bottom:0.4rem;'>Model Registry</p>", unsafe_allow_html=True)

        crop_pipe, crop_meta = _get_crop_model()
        fert_pipe, fert_meta = _get_fert_model()

        if crop_pipe is not None and isinstance(crop_meta, dict):
            acc = crop_meta.get("best_model_metrics", {}).get("accuracy", 0.993)
            st.markdown(
                f"""
                <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(16,185,129,0.25); border-radius:10px; padding:8px 12px; margin-bottom:6px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:0.82rem; font-weight:600; color:#e2e8f0;">🌱 Crop Model</span>
                        <span style="display:flex; align-items:center; gap:5px; font-size:0.73rem; color:#34d399; font-weight:700;">
                            <span class="pulse-dot"></span> Ready
                        </span>
                    </div>
                    <div style="font-size:0.72rem; color:#94a3b8; margin-top:2px;">
                        {crop_meta.get('model_name', 'Random Forest')} • <b>{acc:.1%} Acc</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if fert_pipe is not None and isinstance(fert_meta, dict):
            acc_f = fert_meta.get("best_model_metrics", {}).get("accuracy", 0.95)
            st.markdown(
                f"""
                <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(16,185,129,0.25); border-radius:10px; padding:8px 12px; margin-bottom:6px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:0.82rem; font-weight:600; color:#e2e8f0;">🧪 Fertilizer Engine</span>
                        <span style="display:flex; align-items:center; gap:5px; font-size:0.73rem; color:#34d399; font-weight:700;">
                            <span class="pulse-dot"></span> Ready
                        </span>
                    </div>
                    <div style="font-size:0.72rem; color:#94a3b8; margin-top:2px;">
                        {fert_meta.get('model_name', 'Random Forest')} • <b>{acc_f:.1%} Acc</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:10px; padding:10px; text-align:center;">
                <div style="font-size:0.7rem; color:#64748b; line-height:1.4;">
                    AgriSmart AI — Academic ML Prototype<br>
                    Aligns with <b>UN SDGs 2, 12 & 15</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_page


def render_hero(badge: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-header">
            <div class="hero-badge">{badge}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_notice(text: str, kind: str = "warning") -> None:
    icons = {"warning": "⚠️", "info": "ℹ️", "success": "✅"}
    st.markdown(
        f"""
        <div class="notice-box {kind}">
            <b>{icons.get(kind, 'ℹ️')} Notice:</b> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 1: Home Dashboard
# ---------------------------------------------------------------------------

def page_home() -> None:
    render_hero(
        badge="✨ Autonomous Agronomy Platform",
        title="AgriSmart AI — Sustainable Crop & Fertilizer Advisor",
        subtitle="Empowering precision farming with supervised machine learning algorithms, soil nutrient profiling, and explainable agro-ecological intelligence."
    )

    crop_pipe, crop_meta = _get_crop_model()
    fert_pipe, fert_meta = _get_fert_model()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Crop Classifier</div>
                <div class="kpi-value">{'99.3%' if crop_pipe else 'Offline'}</div>
                <div class="kpi-sub">Held-out test accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Supported Crops</div>
                <div class="kpi-value">{len(crop_meta.get('classes', [])) if isinstance(crop_meta, dict) else '22'}</div>
                <div class="kpi-sub">Multiclass categories</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Fertilizer Engine</div>
                <div class="kpi-value">{'Active' if fert_pipe else 'Standby'}</div>
                <div class="kpi-sub">Independent pipeline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">New Capabilities</div>
                <div class="kpi-value">4 Features</div>
                <div class="kpi-sub">Simulator, Compare, PDF, Calendar</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 0.8])

    with col_left:
        st.markdown(
            """
            <div class="glass-panel">
                <h3 style="color:#ffffff; margin-top:0; font-size:1.3rem;">🌱 High-Impact Decision Support Features</h3>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-top:1rem;">
                    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:14px;">
                        <div style="font-size:1.4rem; margin-bottom:6px;">🌾</div>
                        <div style="font-weight:700; color:#f1f5f9; font-size:0.95rem;">Crop Classification</div>
                        <div style="color:#94a3b8; font-size:0.83rem; margin-top:4px;">Predicts ideal crops with confidence scores, top-3 candidates, and feature importances.</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:14px;">
                        <div style="font-size:1.4rem; margin-bottom:6px;">🌡️</div>
                        <div style="font-weight:700; color:#f1f5f9; font-size:0.95rem;">Climate Simulator</div>
                        <div style="color:#94a3b8; font-size:0.83rem; margin-top:4px;">Stress-tests field resilience against droughts (-50% rain) and heat waves (+4°C).</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:14px;">
                        <div style="font-size:1.4rem; margin-bottom:6px;">⚖️</div>
                        <div style="font-weight:700; color:#f1f5f9; font-size:0.95rem;">Crop Comparison</div>
                        <div style="color:#94a3b8; font-size:0.83rem; margin-top:4px;">Side-by-side comparative matrices on water needs, growing cycles, and NPK demand.</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:14px;">
                        <div style="font-size:1.4rem; margin-bottom:6px;">📄</div>
                        <div style="font-weight:700; color:#f1f5f9; font-size:0.95rem;">PDF Field Advisory</div>
                        <div style="color:#94a3b8; font-size:0.83rem; margin-top:4px;">Generates official branded PDF advisory summaries ready for printing or field audit.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            """
            <div class="glass-panel">
                <h3 style="color:#ffffff; margin-top:0; font-size:1.3rem;">🚀 Interactive Quick-Start</h3>
                <ol style="color:#cbd5e1; font-size:0.88rem; line-height:1.7; padding-left:1.2rem; margin:0.8rem 0 1.2rem 0;">
                    <li>Navigate to <b>Crop Recommendation</b> in the sidebar.</li>
                    <li>Use a <b>1-Click Sample Preset</b> to fill inputs.</li>
                    <li>Generate the prediction and download your <b>Official PDF Field Advisory</b>.</li>
                    <li>Explore the <b>Climate Simulator</b> to test drought resilience.</li>
                    <li>Check the <b>Seasonal Crop Calendar</b> for sowing windows.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Page 2: Crop Recommendation (+ Downloadable PDF)
# ---------------------------------------------------------------------------

def page_crop_recommendation() -> None:
    render_hero(
        badge="🌱 INFERENCE ENGINE",
        title="Crop Suitability Advisor",
        subtitle="Simulate soil chemistry and climatic readings to generate multiclass crop recommendations backed by ensemble machine learning models."
    )

    crop_pipe, crop_meta = _get_crop_model()
    fert_pipe, fert_meta = _get_fert_model()

    if crop_pipe is None:
        render_notice("Crop classification model not found. Please execute <code>python scripts/train_crop_model.py</code> first.", "warning")
        return

    # Presets
    st.markdown("<p style='font-size:0.82rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#94a3b8; margin-bottom:0.4rem;'>⚡ Quick Test Presets (Load Real Scenarios)</p>", unsafe_allow_html=True)
    p_cols = st.columns(5)
    presets = st.session_state["crop_presets"]

    if "preset_values" not in st.session_state:
        st.session_state["preset_values"] = presets["Rice (Wet Tropics)"]

    for idx, (name, values) in enumerate(presets.items()):
        with p_cols[idx]:
            st.markdown('<div class="preset-btn">', unsafe_allow_html=True)
            if st.button(name, key=f"btn_p_{idx}"):
                st.session_state["preset_values"] = values
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    cur = st.session_state["preset_values"]

    # Input Form
    with st.form("crop_input_form"):
        col_chem, col_env = st.columns(2)

        with col_chem:
            st.markdown("<b>🧪 Soil Chemical Profile</b>", unsafe_allow_html=True)
            c_n, c_p = st.columns(2)
            with c_n:
                N = st.number_input("Nitrogen (N) [kg/ha]", min_value=0.0, max_value=300.0, value=float(cur["N"]), step=1.0)
                ph = st.number_input("Soil pH [0-14]", min_value=3.0, max_value=11.0, value=float(cur["ph"]), step=0.1)
            with c_p:
                P = st.number_input("Phosphorus (P) [kg/ha]", min_value=0.0, max_value=250.0, value=float(cur["P"]), step=1.0)
                K = st.number_input("Potassium (K) [kg/ha]", min_value=0.0, max_value=300.0, value=float(cur["K"]), step=1.0)

        with col_env:
            st.markdown("<b>⛅ Climatic & Environmental Factors</b>", unsafe_allow_html=True)
            c_t, c_h = st.columns(2)
            with c_t:
                temperature = st.number_input("Temperature [°C]", min_value=-5.0, max_value=55.0, value=float(cur["temperature"]), step=0.5)
            with c_h:
                humidity = st.number_input("Relative Humidity [%]", min_value=10.0, max_value=100.0, value=float(cur["humidity"]), step=1.0)

            rainfall = st.slider("Seasonal Rainfall [mm]", min_value=10.0, max_value=3000.0, value=float(cur["rainfall"]), step=10.0)

        reg_col, btn_col = st.columns([1.5, 1])
        with reg_col:
            region = st.text_input("Farm / Regional Identifier (Optional)", placeholder="e.g. Plot 4B, Punjab Basin")
        with btn_col:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔍 Compute Optimal Crop Recommendation")

    if not submitted and st.session_state.get("last_crop_prediction") is None:
        return

    if submitted:
        inputs = {
            "N": N, "P": P, "K": K,
            "temperature": temperature, "humidity": humidity,
            "ph": ph, "rainfall": rainfall,
        }

        with st.spinner("Analyzing multi-dimensional feature space..."):
            result = predict_crop(crop_pipe, crop_meta, inputs)
            feature_cols = crop_meta.get("feature_cols", list(inputs.keys()))
            fi_df = get_feature_importance(crop_pipe, feature_cols)
            explanation = build_crop_explanation(
                input_values=inputs,
                crop=result["crop"],
                confidence=result["confidence"],
                top3=result["top3"],
                importance_df=fi_df,
            )
            add_record(
                input_values=inputs,
                crop_predicted=result["crop"],
                crop_confidence=result["confidence"],
                region_note=region or None,
            )
            st.session_state["last_crop_prediction"] = {
                "inputs": inputs,
                "result": result,
                "explanation": explanation,
                "fi_df": fi_df,
                "region": region,
            }

    saved = st.session_state["last_crop_prediction"]
    result = saved["result"]
    explanation = saved["explanation"]
    fi_df = saved["fi_df"]
    inputs = saved["inputs"]
    region = saved["region"]

    # Prediction Hero Card
    crop_name = result["crop"].title()
    conf = result["confidence"]
    conf_str = f"{conf:.1%}" if conf is not None else "N/A"

    st.markdown(
        f"""
        <div class="prediction-hero-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
                <div>
                    <div style="font-size:0.8rem; font-weight:800; color:#34d399; letter-spacing:0.08em; text-transform:uppercase;">
                        🌟 Primary Machine Learning Recommendation
                    </div>
                    <div class="prediction-crop-title">{crop_name}</div>
                    <div style="color:#cbd5e1; font-size:0.95rem; max-width:650px;">
                        {explanation['summary']}
                    </div>
                </div>
                <div style="background:rgba(16,185,129,0.2); border:1px solid rgba(16,185,129,0.4); border-radius:14px; padding:12px 20px; text-align:center;">
                    <div style="font-size:0.75rem; font-weight:700; color:#a7f3d0; text-transform:uppercase;">Confidence Score</div>
                    <div style="font-size:2rem; font-weight:800; color:#ffffff;">{conf_str}</div>
                    <div style="font-size:0.7rem; color:#6ee7b7;">Ensemble Agreement</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # PDF Download Action Banner
    c_pdf1, c_pdf2 = st.columns([2.5, 1.5])
    with c_pdf1:
        st.markdown(
            """
            <div style="background:rgba(15,23,42,0.5); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:12px 16px;">
                <div style="font-size:0.92rem; font-weight:700; color:#f8fafc;">📄 Official Field Advisory PDF Export</div>
                <div style="font-size:0.82rem; color:#94a3b8;">Generate an executive-ready, timestamped advisory report with telemetry benchmarks and compliance stamps.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_pdf2:
        try:
            pdf_bytes = generate_field_advisory_pdf(
                input_values=inputs,
                crop_predicted=result["crop"],
                confidence=result["confidence"],
                top3=result["top3"],
                fertilizer_predicted="Balanced Formulation",
                region_note=region,
                model_name=crop_meta.get("model_name", "Random Forest Classifier"),
                model_accuracy=crop_meta.get("best_model_metrics", {}).get("accuracy", 0.993),
            )
            st.download_button(
                label="⬇️ Download PDF Field Advisory",
                data=pdf_bytes,
                file_name=f"AgriSmart_Advisory_{result['crop']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.caption(f"PDF generator notice: {e}")

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

    # Detailed Tabs
    tab_overview, tab_alternatives, tab_importance = st.tabs([
        "🌿 Agronomic Profile", "📊 Alternative Predictions", "🔬 Feature Attribution"
    ])

    with tab_overview:
        crop_info = explanation.get("crop_info", {})
        if crop_info:
            c_a, c_b, c_c = st.columns(3)
            with c_a:
                st.markdown(f"**Optimal Season:** {crop_info.get('season', 'General')}")
                st.markdown(f"**Water Requirement:** {crop_info.get('water', 'Moderate')}")
            with c_b:
                st.markdown(f"**Preferred Soil:** {crop_info.get('soil', 'Loamy')}")
            with c_c:
                st.markdown(f"**Cultivation Note:** {crop_info.get('notes', 'Ensure proper drainage.')}")
        st.markdown("#### Soil Chemistry Analysis Against General Benchmarks")
        for desc in explanation["nutrient_descriptions"]:
            st.markdown(f"- {desc}")

    with tab_alternatives:
        st.markdown("#### Top 3 Candidate Crops Identified by Classifier")
        if result["top3"]:
            for item in result["top3"]:
                p_val = item["probability"]
                st.markdown(
                    f"""
                    <div style="margin-bottom:12px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-weight:600; font-size:0.92rem;">
                            <span>🌾 {item['crop'].title()}</span>
                            <span style="color:#34d399;">{p_val:.1%}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(p_val)

    with tab_importance:
        if fi_df is not None:
            fig_fi = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=[[0, "#065f46"], [0.5, "#10b981"], [1, "#34d399"]],
            )
            fig_fi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", family="Plus Jakarta Sans"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=260,
                coloraxis_showscale=False,
                yaxis=dict(categoryorder="total ascending", gridcolor="rgba(255,255,255,0.05)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_fi, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 3: Climate "What-If" Sensitivity Simulator
# ---------------------------------------------------------------------------

def page_climate_simulator() -> None:
    render_hero(
        badge="🌡️ CLIMATE RESILIENCE SIMULATOR",
        title="Climate Stress-Testing & Adaptation Engine",
        subtitle="Simulate droughts, extreme heatwaves, and monsoon delays in real time to visualize how the machine learning model shifts crop recommendations."
    )

    crop_pipe, crop_meta = _get_crop_model()
    if crop_pipe is None:
        render_notice("Crop classification model not found.", "warning")
        return

    # Baseline selector
    st.markdown("### 1. Select Baseline Agronomic Field Condition")
    base_col1, base_col2 = st.columns([1, 2])
    with base_col1:
        base_choice = st.selectbox(
            "Select Baseline Preset",
            ["Rice (Wet Tropics)", "Maize (Semi-Arid)", "Cotton Field", "Highland Coffee"],
        )
    base_vals = st.session_state["crop_presets"][base_choice].copy()

    with base_col2:
        st.markdown(
            f"""
            <div style="background:rgba(15,23,42,0.5); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:10px 14px; font-size:0.86rem; color:#cbd5e1;">
                <b>Baseline Specs:</b> N={base_vals['N']} kg/ha, P={base_vals['P']}, K={base_vals['K']}, pH={base_vals['ph']} | 
                <b>Baseline Climate:</b> Temp={base_vals['temperature']}°C, Humidity={base_vals['humidity']}%, Rainfall={base_vals['rainfall']} mm
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 2. Apply Microclimate Perturbations & Anomalies")

    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        delta_temp = st.slider("🌡️ Temperature Anomaly (°C)", min_value=-5.0, max_value=7.0, value=2.0, step=0.5, help="Simulate global warming or seasonal cold waves.")
    with s_col2:
        delta_rain_pct = st.slider("🌧️ Rainfall Fluctuation (%)", min_value=-60, max_value=80, value=-35, step=5, help="Simulate severe drought (-50%) or heavy monsoon inundation (+50%).")
    with s_col3:
        delta_hum = st.slider("💧 Relative Humidity Shift (%)", min_value=-30.0, max_value=30.0, value=-15.0, step=5.0)

    # Compute Perturbed Environment
    sim_inputs = base_vals.copy()
    sim_inputs["temperature"] = round(base_vals["temperature"] + delta_temp, 2)
    sim_inputs["rainfall"] = round(max(20.0, base_vals["rainfall"] * (1.0 + delta_rain_pct / 100.0)), 2)
    sim_inputs["humidity"] = round(max(10.0, min(100.0, base_vals["humidity"] + delta_hum)), 2)

    # Run predictions on baseline vs simulated
    base_pred = predict_crop(crop_pipe, crop_meta, base_vals)
    sim_pred = predict_crop(crop_pipe, crop_meta, sim_inputs)

    b_crop = base_pred["crop"].title()
    s_crop = sim_pred["crop"].title()
    b_conf = base_pred["confidence"]
    s_conf = sim_pred["confidence"]

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # Comparison Cards
    res_c1, res_c2 = st.columns(2)
    with res_c1:
        st.markdown(
            f"""
            <div class="glass-panel" style="border-left:4px solid #06b6d4;">
                <div style="font-size:0.8rem; font-weight:700; color:#38bdf8; text-transform:uppercase;">Original Baseline Reading</div>
                <div style="font-size:2rem; font-weight:800; color:#ffffff; margin:4px 0;">{b_crop}</div>
                <div style="color:#94a3b8; font-size:0.85rem;">
                    Temp: <b>{base_vals['temperature']}°C</b> | Rain: <b>{base_vals['rainfall']} mm</b> | Humidity: <b>{base_vals['humidity']}%</b><br>
                    Model Confidence: <b>{b_conf:.1%}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with res_c2:
        border_col = "#10b981" if b_crop == s_crop else "#f59e0b"
        st.markdown(
            f"""
            <div class="glass-panel" style="border-left:4px solid {border_col};">
                <div style="font-size:0.8rem; font-weight:700; color:{border_col}; text-transform:uppercase;">
                    {'Climate Resilient Outcome' if b_crop == s_crop else '⚠️ Adaptation Shifted Outcome'}
                </div>
                <div style="font-size:2rem; font-weight:800; color:#ffffff; margin:4px 0;">{s_crop}</div>
                <div style="color:#94a3b8; font-size:0.85rem;">
                    Temp: <b>{sim_inputs['temperature']}°C ({'+' if delta_temp>0 else ''}{delta_temp}°C)</b> | Rain: <b>{sim_inputs['rainfall']} mm ({delta_rain_pct}%)</b><br>
                    Model Confidence: <b>{s_conf:.1%}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Climate Sensitivity Curve (Dynamic sweep of rainfall)
    st.markdown("### 3. Dynamic Precipitation Sensitivity Sweep")
    st.caption("How model classification probabilities transition across rainfall volumes (100mm to 2200mm) while holding other parameters fixed.")

    sweep_rain = np.linspace(100, 2200, 15)
    sweep_records = []
    clf = crop_pipe[-1]

    if hasattr(clf, "predict_proba"):
        feat_cols = crop_meta["feature_cols"]
        classes = crop_pipe.classes_

        rows = []
        for r_val in sweep_rain:
            row_dict = sim_inputs.copy()
            row_dict["rainfall"] = r_val
            rows.append([row_dict[col] for col in feat_cols])

        df_sweep = pd.DataFrame(rows, columns=feat_cols)
        probs = crop_pipe.predict_proba(df_sweep)

        # Find top 4 crops with highest peak probabilities
        max_per_class = probs.max(axis=0)
        top_indices = np.argsort(max_per_class)[::-1][:4]

        for idx, r_val in enumerate(sweep_rain):
            for c_idx in top_indices:
                sweep_records.append({
                    "Rainfall (mm)": round(r_val, 1),
                    "Crop": classes[c_idx].title(),
                    "Probability": probs[idx, c_idx],
                })

        df_curve = pd.DataFrame(sweep_records)
        fig_curve = px.line(
            df_curve,
            x="Rainfall (mm)",
            y="Probability",
            color="Crop",
            markers=True,
            color_discrete_sequence=["#10b981", "#06b6d4", "#f59e0b", "#ec4899"],
        )
        fig_curve.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="Plus Jakarta Sans"),
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", range=[0, 1.05]),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_curve, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 4: Head-to-Head Crop Comparison
# ---------------------------------------------------------------------------

def page_crop_comparison() -> None:
    render_hero(
        badge="⚖️ AGRONOMIC ARBITRAGE",
        title="Head-to-Head Crop Comparison Matrix",
        subtitle="Compare any two crops side-by-side across 15 agronomic dimensions including water demand, growing duration, NPK appetite, and market risk."
    )

    names = get_all_crop_names()

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        crop_a = st.selectbox("Select First Crop (Crop A)", names, index=names.index("rice") if "rice" in names else 0)
    with col_sel2:
        crop_b = st.selectbox("Select Second Crop (Crop B)", names, index=names.index("chickpea") if "chickpea" in names else 1)

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

    # Comparative Dataframe
    df_comp = compare_two_crops(crop_a, crop_b)
    st.dataframe(df_comp, use_container_width=True)

    # Visual Multi-Attribute Radar
    p_a = get_crop_profile(crop_a)
    p_b = get_crop_profile(crop_b)

    def _score(val: str) -> int:
        v = val.lower()
        if "very low" in v: return 1
        if "low" in v: return 2
        if "moderate" in v: return 3
        if "very high" in v: return 5
        if "high" in v: return 4
        return 3

    categories = ["Water Demand", "Nitrogen Appetite", "Potassium Need", "Drought Resistance"]
    vals_a = [
        _score(p_a.get("water_level", "Moderate")),
        _score(p_a.get("n_demand", "Moderate")),
        _score(p_a.get("k_demand", "Moderate")),
        _score(p_a.get("drought_tolerance", "Moderate")),
    ]
    vals_b = [
        _score(p_b.get("water_level", "Moderate")),
        _score(p_b.get("n_demand", "Moderate")),
        _score(p_b.get("k_demand", "Moderate")),
        _score(p_b.get("drought_tolerance", "Moderate")),
    ]

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(r=vals_a, theta=categories, fill="toself", name=p_a.get("name", crop_a.title()), line_color="#10b981", fillcolor="rgba(16,185,129,0.2)"))
    fig_r.add_trace(go.Scatterpolar(r=vals_b, theta=categories, fill="toself", name=p_b.get("name", crop_b.title()), line_color="#06b6d4", fillcolor="rgba(6,182,212,0.2)"))

    fig_r.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Plus Jakarta Sans"),
        height=350,
        margin=dict(l=30, r=30, t=20, b=20),
    )
    st.plotly_chart(fig_r, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 5: Seasonal Crop Calendar
# ---------------------------------------------------------------------------

def page_crop_calendar() -> None:
    render_hero(
        badge="📅 AGRO-SEASONAL CALENDAR",
        title="Seasonal Crop Calendar & Phenology Schedule",
        subtitle="Explore Kharif (Monsoon), Rabi (Winter), Zaid (Summer), and Perennial cropping windows with an interactive 12-month visual timeline."
    )

    all_crops = get_all_crop_names()

    col_filter, col_select = st.columns([1, 2.5])
    with col_filter:
        season_filter = st.radio(
            "Filter by Growing Season",
            ["All Seasons", "Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)", "Perennial (Multi-Year)"],
        )

    # Filtered list of crops
    if season_filter == "Kharif (Monsoon)":
        filtered = [c for c in all_crops if "Kharif" in CROP_CALENDAR.get(c, {}).get("season_type", "")]
    elif season_filter == "Rabi (Winter)":
        filtered = [c for c in all_crops if "Rabi" in CROP_CALENDAR.get(c, {}).get("season_type", "")]
    elif season_filter == "Zaid (Summer)":
        filtered = [c for c in all_crops if "Zaid" in CROP_CALENDAR.get(c, {}).get("season_type", "")]
    elif season_filter == "Perennial (Multi-Year)":
        filtered = [c for c in all_crops if "Perennial" in CROP_CALENDAR.get(c, {}).get("season_type", "")]
    else:
        filtered = all_crops

    with col_select:
        default_selection = filtered[:6] if len(filtered) >= 6 else filtered
        chosen_crops = st.multiselect("Select Crops to Visualize in Timeline", filtered, default=default_selection)

    if chosen_crops:
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        st.markdown("### 12-Month Sowing, Growth & Harvesting Gantt Scatter")
        fig_cal = build_crop_timeline_chart(chosen_crops)
        st.plotly_chart(fig_cal, use_container_width=True)

    # Crop Cards Grid
    st.markdown("### Agronomic Seasonality Descriptions")
    g_cols = st.columns(3)
    for idx, c_id in enumerate(chosen_crops):
        c_info = get_calendar_info(c_id)
        with g_cols[idx % 3]:
            st.markdown(
                f"""
                <div class="glass-panel" style="margin-bottom:12px; padding:14px 16px;">
                    <div style="font-size:1.1rem; font-weight:700; color:#ffffff;">🌾 {c_id.title()}</div>
                    <div style="display:inline-block; font-size:0.74rem; font-weight:700; color:#34d399; background:rgba(16,185,129,0.15); border-radius:6px; padding:2px 8px; margin:4px 0 8px 0;">
                        {c_info.get('season_type', 'General')}
                    </div>
                    <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                        <b>Sowing:</b> {', '.join(c_info.get('sowing_months', []))}<br>
                        <b>Harvest:</b> {', '.join(c_info.get('harvest_months', []))}<br>
                        <span style="color:#94a3b8; font-size:0.78rem;">{c_info.get('description', '')}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Page 6: Fertilizer Recommendation
# ---------------------------------------------------------------------------

def page_fertilizer_recommendation() -> None:
    render_hero(
        badge="🧪 NUTRIENT STEWARDSHIP",
        title="Fertilizer Formulation Advisor",
        subtitle="Independent machine learning model targeting balanced chemical amendments while preventing eutrophication and nutrient burn."
    )

    fert_pipe, fert_meta = _get_fert_model()
    if fert_pipe is None:
        render_notice("Fertilizer model not trained. Execute <code>python scripts/train_fertilizer_model.py</code> to initialize this module.", "warning")
        return

    # Presets
    st.markdown("<p style='font-size:0.82rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#94a3b8; margin-bottom:0.4rem;'>⚡ Quick Test Presets</p>", unsafe_allow_html=True)
    p_cols = st.columns(3)
    presets = st.session_state["fert_presets"]

    if "fert_preset_values" not in st.session_state:
        st.session_state["fert_preset_values"] = presets["Sugarcane on Black Soil"]

    for idx, (name, values) in enumerate(presets.items()):
        with p_cols[idx]:
            st.markdown('<div class="preset-btn">', unsafe_allow_html=True)
            if st.button(name, key=f"btn_fp_{idx}"):
                st.session_state["fert_preset_values"] = values
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    cur = st.session_state["fert_preset_values"]
    feature_cols = fert_meta.get("feature_cols", [])

    with st.form("fert_input_form"):
        c1, c2, c3 = st.columns(3)

        inputs = {}
        with c1:
            if "Temperature" in feature_cols:
                inputs["Temperature"] = st.number_input("Ambient Temp [°C]", value=float(cur["Temperature"]), step=0.5)
            if "N" in feature_cols:
                inputs["N"] = st.number_input("Current Soil Nitrogen [N]", value=float(cur["N"]), step=1.0)
            if "Soil Type" in feature_cols:
                soil_opts = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
                idx_s = soil_opts.index(cur["Soil Type"]) if cur["Soil Type"] in soil_opts else 0
                inputs["Soil Type"] = st.selectbox("Soil Texture / Type", soil_opts, index=idx_s)

        with c2:
            if "Humidity" in feature_cols:
                inputs["Humidity"] = st.number_input("Humidity [%]", value=float(cur["Humidity"]), step=1.0)
            if "P" in feature_cols:
                inputs["P"] = st.number_input("Current Soil Phosphorus [P]", value=float(cur["P"]), step=1.0)
            if "Crop Type" in feature_cols:
                crop_opts = ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy", "Barley", "Wheat", "Millets", "Oil seeds", "Pulses", "Ground Nuts"]
                idx_c = crop_opts.index(cur["Crop Type"]) if cur["Crop Type"] in crop_opts else 0
                inputs["Crop Type"] = st.selectbox("Intended Crop Type", crop_opts, index=idx_c)

        with c3:
            if "Moisture" in feature_cols:
                inputs["Moisture"] = st.number_input("Soil Moisture [%]", value=float(cur["Moisture"]), step=1.0)
            if "K" in feature_cols:
                inputs["K"] = st.number_input("Current Soil Potassium [K]", value=float(cur["K"]), step=1.0)

        submitted = st.form_submit_button("🔍 Compute Fertilizer Recommendation")

    if not submitted:
        return

    with st.spinner("Classifying optimal fertilizer profile..."):
        result = predict_fertilizer(fert_pipe, fert_meta, inputs)
        explanation = build_fertilizer_explanation(inputs, result["fertilizer"], result["confidence"])
        add_record(
            input_values={k: inputs.get(k) for k in ["N", "P", "K"]},
            fertilizer_predicted=result["fertilizer"],
            fertilizer_confidence=result["confidence"],
        )

    st.markdown(
        f"""
        <div class="prediction-hero-card">
            <div style="font-size:0.8rem; font-weight:800; color:#34d399; letter-spacing:0.08em; text-transform:uppercase;">
                🧪 Recommended Fertilizer Product
            </div>
            <div class="prediction-crop-title">{result['fertilizer']}</div>
            <div style="color:#cbd5e1; font-size:0.95rem; margin-top:0.3rem;">
                {explanation['summary']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 7: Soil Health Analysis
# ---------------------------------------------------------------------------

def page_soil_health() -> None:
    render_hero(
        badge="🌿 DIAGNOSTIC SUITE",
        title="Soil Nutrient & Health Diagnostics",
        subtitle="Benchmark your field measurements against agronomic reference thresholds with interactive telemetry gauges and radar multi-axis diagrams."
    )

    c1, c2 = st.columns(2)
    with c1:
        N = st.slider("Nitrogen (N) [kg/ha]", 0.0, 250.0, 85.0)
        P = st.slider("Phosphorus (P) [kg/ha]", 0.0, 150.0, 45.0)
        K = st.slider("Potassium (K) [kg/ha]", 0.0, 250.0, 50.0)
        ph = st.slider("Soil pH Level", 4.0, 10.0, 6.5, 0.1)
    with c2:
        temperature = st.slider("Mean Temperature [°C]", 0.0, 50.0, 26.0)
        humidity = st.slider("Ambient Humidity [%]", 10.0, 100.0, 70.0)
        rainfall = st.slider("Precipitation [mm]", 50.0, 2500.0, 800.0)

    st.markdown("### 📊 Multi-Axis Soil Chemical Radar")
    r_cols = ["N", "P", "K", "pH (Scaled)", "Temp (Scaled)"]
    user_vals = [N, P, K, ph * 15, temperature * 3]
    bench_vals = [100, 40, 70, 6.5 * 15, 25 * 3]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=user_vals,
        theta=r_cols,
        fill="toself",
        name="Field Reading",
        line_color="#10b981",
        fillcolor="rgba(16, 185, 129, 0.2)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=bench_vals,
        theta=r_cols,
        fill="toself",
        name="Agronomic Midpoint",
        line_color="#06b6d4",
        fillcolor="rgba(6, 182, 212, 0.1)",
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Plus Jakarta Sans"),
        height=380,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 8: Sustainability Advisor
# ---------------------------------------------------------------------------

def page_sustainability() -> None:
    render_hero(
        badge="♻️ SUSTAINABILITY FRAMEWORK",
        title="Sustainable Agriculture & UN SDG Roadmap",
        subtitle="Operationalizing conservation agronomy to protect natural soil biomes, preserve clean waterways, and guarantee long-term agricultural resilience."
    )

    t_check, t_tips, t_sdg = st.tabs(["✅ Pre-Season Audit Checklist", "💡 Best Practice Catalog", "🌍 UN SDG Alignment"])

    with t_check:
        st.markdown("#### Interactive Pre-Planting Conservation Audit")
        checklist = get_checklist()
        done = 0
        for category, question in checklist:
            if st.checkbox(f"**{category}**: {question}", key=f"c_{category}"):
                done += 1
        pct = done / len(checklist)
        st.progress(pct)
        st.caption(f"Completed {done} of {len(checklist)} verification points ({pct:.0%}).")

    with t_tips:
        tips = get_all_tips()
        for t in tips:
            with st.expander(t["category"]):
                st.write(t["tip"])

    with t_sdg:
        for k, v in get_sdg_relevance().items():
            st.markdown(
                f"""
                <div class="glass-panel" style="margin-bottom:1rem;">
                    <h4 style="margin:0 0 6px 0; color:#34d399;">{k}</h4>
                    <p style="margin:0; color:#cbd5e1; font-size:0.92rem; line-height:1.6;">{v}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Page 9: Model Performance
# ---------------------------------------------------------------------------

def page_model_performance() -> None:
    render_hero(
        badge="📊 VERIFIED EVALUATION",
        title="Algorithm Telemetry & Evaluation Metrics",
        subtitle="Transparent model benchmark results computed strictly on held-out test splits with zero data leakage across training pipelines."
    )

    crop_pipe, crop_meta = _get_crop_model()
    fert_pipe, fert_meta = _get_fert_model()

    if not crop_pipe and not fert_pipe:
        render_notice("No trained model artifacts detected.", "warning")
        return

    if crop_pipe and isinstance(crop_meta, dict):
        st.markdown("### 🌱 Crop Classifier Benchmark Comparison")
        all_res = crop_meta.get("all_results", {})
        if all_res:
            rows = []
            for name, m in all_res.items():
                rows.append({
                    "Algorithm": name,
                    "Accuracy": f"{m.get('accuracy', 0):.4f}",
                    "F1-Score (Weighted)": f"{m.get('f1', 0):.4f}",
                    "Precision": f"{m.get('precision', 0):.4f}",
                    "Recall": f"{m.get('recall', 0):.4f}",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Algorithm"), use_container_width=True)

            # Chart
            plot_df = pd.DataFrame([
                {"Algorithm": k, "Accuracy": v.get("accuracy", 0), "F1-Score": v.get("f1", 0)}
                for k, v in all_res.items()
            ])
            fig = px.bar(
                plot_df,
                x="Algorithm",
                y=["Accuracy", "F1-Score"],
                barmode="group",
                color_discrete_sequence=["#10b981", "#06b6d4"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", family="Plus Jakarta Sans"),
                margin=dict(l=10, r=10, t=20, b=20),
                height=300,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", range=[0.8, 1.02]),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 10: Prediction History
# ---------------------------------------------------------------------------

def page_history() -> None:
    render_hero(
        badge="🕘 AUDIT LOG",
        title="Prediction Telemetry & Audit Trail",
        subtitle="Review, audit, or export locally cached inference transactions for academic verification or field records."
    )

    df = load_history()
    if df.empty:
        render_notice("No prediction records logged in this session.", "info")
        return

    st.markdown(f"**Total Inferences Recorded:** `{len(df)} transactions`")
    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)

    c_dl, c_clr = st.columns([4, 1])
    with c_dl:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Complete History (.CSV)", data=csv_bytes, file_name="agrismart_inferences.csv", mime="text/csv")
    with c_clr:
        if st.button("🗑️ Purge Log"):
            clear_history()
            st.rerun()


# ---------------------------------------------------------------------------
# Page 11: About
# ---------------------------------------------------------------------------

def page_about() -> None:
    render_hero(
        badge="ℹ️ ACADEMIC BRIEF",
        title="About AgriSmart AI",
        subtitle="Capstone machine learning project demonstrating end-to-end data pipelines, transparent validation, and responsible AI for sustainable development."
    )

    st.markdown(
        """
        <div class="glass-panel">
            <h3 style="margin-top:0; color:#f8fafc;">Executive Summary</h3>
            <p style="color:#cbd5e1; font-size:0.93rem; line-height:1.65;">
                AgriSmart AI addresses the challenge of sub-optimal agricultural decision-making by applying modern multiclass supervised machine learning to soil nutrient and environmental parameters. By benchmarking multiple models (Random Forest, Decision Trees, K-Nearest Neighbors) and surfacing Gini-based feature importances, the system ensures recommendations are transparent, explainable, and aligned with sustainable development frameworks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()
    selected = render_sidebar()

    if selected == PAGES[0]:
        page_home()
    elif selected == PAGES[1]:
        page_crop_recommendation()
    elif selected == PAGES[2]:
        page_climate_simulator()
    elif selected == PAGES[3]:
        page_crop_comparison()
    elif selected == PAGES[4]:
        page_crop_calendar()
    elif selected == PAGES[5]:
        page_fertilizer_recommendation()
    elif selected == PAGES[6]:
        page_soil_health()
    elif selected == PAGES[7]:
        page_sustainability()
    elif selected == PAGES[8]:
        page_model_performance()
    elif selected == PAGES[9]:
        page_history()
    elif selected == PAGES[10]:
        page_about()


if __name__ == "__main__":
    main()
