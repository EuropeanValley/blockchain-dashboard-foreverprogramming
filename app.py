"""
CryptoChain Analyzer Dashboard
================================
Entry point: streamlit run app.py

Modules:
  M1 – Proof of Work Monitor      (live difficulty, hash rate, block times)
  M2 – Block Header Analyzer      (80-byte header, local PoW verification)
  M3 – Difficulty History         (adjustment epochs, block-time ratio)
  M4 – AI Fee Estimator           (Gradient Boosting, sat/vByte prediction)

Student: Zihao Ying | GitHub: foreverprogramming
Course:  Cryptography – Universidad Alfonso X el Sabio
Prof:    Jorge Calvo | AY 2025-26
"""

import streamlit as st

st.set_page_config(
    page_title="CryptoChain Analyzer",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for a cleaner look ─────────────────────────────────────────
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"]      { font-size: 0.95rem; font-weight: 600; }
    div[data-testid="metric-container"] > div { font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────
st.title("₿ CryptoChain Analyzer Dashboard")
st.caption(
    "Real-time Bitcoin cryptographic metrics | "
    "Cryptography — UAX | Prof. Jorge Calvo | AY 2025-26"
)

from modules.m1_pow_monitor import render as render_m1
from modules.m2_block_header import render as render_m2
from modules.m3_difficulty_history import render as render_m3
from modules.m4_ai_component import render as render_m4

tab1, tab2, tab3, tab4 = st.tabs([
    "⛏️ M1 · PoW Monitor",
    "🔍 M2 · Block Header",
    "📈 M3 · Difficulty History",
    "🤖 M4 · Fee Estimator (AI)",
])

with tab1:
    render_m1()

with tab2:
    render_m2()

with tab3:
    render_m3()

with tab4:
    render_m4()
