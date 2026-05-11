"""
CryptoChain Analyzer Dashboard
Student: Zihao Ying | GitHub: foreverprogramming
Cryptography – UAX | Prof. Jorge Calvo | AY 2025-26
"""

import time
from datetime import datetime, timezone

import streamlit as st

st.set_page_config(
    page_title="CryptoChain Analyzer",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
:root {
    --btc:#f7931a; --btc-dim:#b86a10; --bg:#0a0c10; --bg2:#111520;
    --bg3:#181d2a; --border:#1e2535; --text:#e2e8f0; --text-dim:#64748b;
    --green:#22c55e; --red:#ef4444; --blue:#3b82f6;
}
.stApp,[data-testid="stAppViewContainer"]>.main{background:var(--bg);}
[data-testid="stHeader"]{background:transparent;}
section[data-testid="stSidebar"]{background:var(--bg2);border-right:1px solid var(--border);}
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;color:var(--text);}
[data-testid="stSidebarContent"]{padding-top:1rem;}
.kpi-card{background:var(--bg3);border:1px solid var(--border);border-radius:10px;
          padding:1rem 1.25rem;margin-bottom:.5rem;}
.kpi-label{font-size:.70rem;font-weight:700;letter-spacing:.10em;text-transform:uppercase;
           color:var(--text-dim);margin-bottom:.3rem;}
.kpi-value{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;font-weight:600;
           color:var(--btc);line-height:1.1;}
.kpi-value.green{color:var(--green);}.kpi-value.blue{color:var(--blue);}
.kpi-value.dim{color:var(--text);}
.kpi-sub{font-size:.72rem;color:var(--text-dim);margin-top:.2rem;}
.panel{background:var(--bg3);border:1px solid var(--border);border-radius:12px;
       padding:1.25rem 1.5rem;margin-bottom:1rem;}
.panel-title{font-size:.78rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
             color:var(--btc);margin-bottom:1rem;padding-bottom:.5rem;
             border-bottom:1px solid var(--border);}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.68rem;
      font-weight:700;letter-spacing:.06em;text-transform:uppercase;}
.pill-live{background:#052e16;color:var(--green);border:1px solid #166534;}
.pill-demo{background:#431407;color:#fb923c;border:1px solid #9a3412;}
.stCodeBlock code,code{font-family:'IBM Plex Mono',monospace!important;
    font-size:.78rem!important;background:#0d1117!important;}
.stButton>button{background:var(--btc);color:#000;font-weight:700;font-size:.80rem;
    letter-spacing:.05em;border:none;border-radius:6px;padding:.45rem 1.1rem;transition:opacity .15s;}
.stButton>button:hover{opacity:.85;}
.stTextInput input{background:#0d1117;border:1px solid var(--border);color:var(--text);
    border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:.82rem;}
.stTextInput input:focus{border-color:var(--btc);box-shadow:none;}
[data-testid="metric-container"]{background:var(--bg3)!important;
    border:1px solid var(--border)!important;border-radius:10px!important;padding:.75rem 1rem!important;}
[data-testid="metric-container"] label{font-size:.70rem!important;font-weight:700!important;
    letter-spacing:.10em!important;text-transform:uppercase!important;color:var(--text-dim)!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{
    font-family:'IBM Plex Mono',monospace!important;font-size:1.25rem!important;color:var(--btc)!important;}
hr{border-color:var(--border)!important;margin:1.25rem 0!important;}
.streamlit-expanderHeader{background:var(--bg3)!important;border:1px solid var(--border)!important;
    border-radius:8px!important;font-size:.80rem!important;font-weight:600!important;color:var(--text-dim)!important;}
.streamlit-expanderContent{border:1px solid var(--border)!important;border-top:none!important;
    background:var(--bg2)!important;}
.stInfo{background:#0c1b33!important;border-left:3px solid var(--blue)!important;}
.stWarning{background:#2d1a00!important;border-left:3px solid var(--btc)!important;}
.stSuccess{background:#052e16!important;border-left:3px solid var(--green)!important;}
.stError{background:#2d0a0a!important;border-left:3px solid var(--red)!important;}
.stDataFrame{border:1px solid var(--border)!important;border-radius:8px!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg2);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:.5rem 0 1.5rem;'>
        <div style='font-size:2.5rem;line-height:1;'>₿</div>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:.90rem;font-weight:700;
                    color:#f7931a;letter-spacing:.12em;text-transform:uppercase;margin-top:.4rem;'>
            CryptoChain</div>
        <div style='font-size:.65rem;color:#64748b;letter-spacing:.08em;
                    text-transform:uppercase;margin-top:.15rem;'>Analyzer Dashboard</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div style='font-size:.62rem;font-weight:700;letter-spacing:.12em;
        text-transform:uppercase;color:#64748b;margin-bottom:.4rem;'>Required</div>""",
        unsafe_allow_html=True)

    required_pages = {
        "⛏️  PoW Monitor":       "m1",
        "🔍  Block Header":       "m2",
        "📈  Difficulty History": "m3",
        "🤖  Fee Estimator (AI)": "m4",
    }
    optional_pages = {
        "🌿  Merkle Verifier":    "m5",
        "🛡️  Security Score":     "m6",
        "🔬  Anomaly Detector":   "m7",
    }

    if "page" not in st.session_state:
        st.session_state["page"] = "m1"

    for label, key in required_pages.items():
        active = st.session_state["page"] == key
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state["page"] = key
            st.rerun()

    st.markdown("""<div style='font-size:.62rem;font-weight:700;letter-spacing:.12em;
        text-transform:uppercase;color:#64748b;margin:.75rem 0 .4rem;'>Optional</div>""",
        unsafe_allow_html=True)

    for label, key in optional_pages.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state["page"] = key
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    now_utc = datetime.now(tz=timezone.utc)
    st.markdown(f"""
    <div style='font-family:"IBM Plex Mono",monospace;font-size:.75rem;
                color:#64748b;text-align:center;line-height:1.8;'>
        <div style='color:#94a3b8;'>{now_utc.strftime('%Y-%m-%d')}</div>
        <div style='font-size:1.1rem;color:#e2e8f0;'>{now_utc.strftime('%H:%M:%S')} UTC</div>
        <div style='margin-top:.5rem;font-size:.62rem;letter-spacing:.06em;'>BITCOIN MAINNET</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.62rem;color:#475569;text-align:center;line-height:1.7;'>
        Zihao Ying · foreverprogramming<br>UAX Cryptography 2025-26<br>Prof. Jorge Calvo
    </div>""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────
page_meta = {
    "m1": ("⛏️  Proof of Work Monitor",   "Live difficulty, hash rate, and inter-block time distribution",                       "Required · M1"),
    "m2": ("🔍  Block Header Analyzer",    "80-byte header dissection with local SHA-256d verification",                          "Required · M2"),
    "m3": ("📈  Difficulty History",        "Adjustment epochs, block-time ratios, and trend analysis",                            "Required · M3"),
    "m4": ("🤖  Fee Estimator (AI)",        "Gradient Boosting · MAE, RMSE, MAPE, R², cross-validation, fee tiers",               "Required · M4"),
    "m5": ("🌿  Merkle Proof Verifier",     "Step-by-step SHA-256d proof of transaction inclusion in a block",                     "Optional · M5"),
    "m6": ("🛡️  Security Score",            "USD/hour cost of a 51% attack · Nakamoto (2008) §11 confirmation depth",             "Optional · M6"),
    "m7": ("🔬  Block Anomaly Detector",    "Exponential baseline · Z-score & p-value flagging · comparison with M4",             "Optional · M7"),
}
title, subtitle, badge = page_meta[st.session_state["page"]]
badge_color = "#1e3a5f" if "Required" in badge else "#1a2e1a"
badge_text_color = "#60a5fa" if "Required" in badge else "#4ade80"

st.markdown(f"""
<div style='padding:.25rem 0 1.25rem;display:flex;align-items:flex-start;gap:1rem;'>
    <div style='flex:1;'>
        <h1 style='font-family:"IBM Plex Sans",sans-serif;font-size:1.55rem;
                   font-weight:700;color:#e2e8f0;margin:0;line-height:1.2;'>{title}</h1>
        <p style='font-size:.80rem;color:#64748b;margin:.3rem 0 0;'>{subtitle}</p>
    </div>
    <div style='padding:.25rem .75rem;background:{badge_color};border-radius:6px;
                font-size:.65rem;font-weight:700;letter-spacing:.08em;
                text-transform:uppercase;color:{badge_text_color};white-space:nowrap;
                margin-top:.25rem;'>{badge}</div>
</div>
""", unsafe_allow_html=True)

# ── Route ─────────────────────────────────────────────────────────────────
page = st.session_state["page"]
if   page == "m1": from modules.m1_pow_monitor       import render; render()
elif page == "m2": from modules.m2_block_header       import render; render()
elif page == "m3": from modules.m3_difficulty_history import render; render()
elif page == "m4": from modules.m4_ai_component       import render; render()
elif page == "m5": from modules.m5_merkle_verifier    import render; render()
elif page == "m6": from modules.m6_security_score     import render; render()
elif page == "m7": from modules.m7_anomaly_detector   import render; render()
