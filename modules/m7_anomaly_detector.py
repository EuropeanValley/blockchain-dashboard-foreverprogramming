"""
M7 – Block Anomaly Detector  (second AI approach)
==================================================
Identifies blocks whose inter-arrival time is statistically abnormal using
an Exponential distribution baseline — a different approach from M4's
supervised regression.

Theory
------
Under the Poisson process model of Bitcoin mining, inter-block times T follow:

    T ~ Exp(λ)    where  λ = 1/600  (one block per 600 s on average)

CDF: P(T ≤ t) = 1 − e^{−λt}
PDF: f(t)      = λ e^{−λt}

An anomalous block is one whose inter-arrival time is in the far tail of
this distribution — i.e. improbably fast OR improbably slow.

Detection method
----------------
We use two complementary tests per block:

1. **Z-score on log-transformed times:**
   log(T) ~ Gumbel(μ, β) — approximately normal for large samples.
   |z| > threshold  →  anomaly flag.

2. **Two-sided p-value:**
   p = 2 × min(F(t), 1−F(t))   where F is the Exp(λ) CDF.
   p < α (e.g. 0.05)  →  anomaly flag.

Comparison with M4
------------------
| Aspect       | M4 – Fee Estimator          | M7 – Anomaly Detector        |
|---|---|---|
| Problem type | Supervised regression        | Unsupervised / statistical   |
| Target       | Continuous fee value         | Binary anomaly label         |
| Model        | Gradient Boosting            | Exponential distribution fit |
| Evaluation   | MAE, RMSE, R²               | Precision, recall, F1        |
| Training     | Block-level feature matrix   | Only inter-arrival times     |

Reference: Nakamoto (2008) §11 (Poisson block timing model).
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from api.blockchain_client import (
    get_recent_blocks, get_tip_height,
    mock_recent_blocks,
)

PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#94a3b8", size=11),
    margin=dict(t=36, b=36, l=8, r=8),
    xaxis=dict(gridcolor="#1e2535", linecolor="#1e2535", zeroline=False),
    yaxis=dict(gridcolor="#1e2535", linecolor="#1e2535", zeroline=False),
)

TARGET_BLOCK_TIME = 600.0   # seconds


# ── Statistical detection ────────────────────────────────────────────────────

def compute_anomalies(blocks: list[dict],
                      z_threshold: float = 2.5,
                      p_threshold: float = 0.05) -> pd.DataFrame:
    """
    Given a list of block dicts (must have 'timestamp' and 'height'),
    compute inter-arrival times and flag anomalies.

    Returns a DataFrame with columns:
        height, timestamp, inter_time_s, z_score, p_value,
        z_anomaly, p_anomaly, anomaly (either flag)
    """
    sorted_blocks = sorted(blocks, key=lambda b: b["height"])
    rows = []
    for i in range(1, len(sorted_blocks)):
        dt = sorted_blocks[i]["timestamp"] - sorted_blocks[i-1]["timestamp"]
        rows.append({
            "height":        sorted_blocks[i]["height"],
            "timestamp":     sorted_blocks[i]["timestamp"],
            "inter_time_s":  float(dt),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ── Exponential CDF: two-sided p-value ──────────────────────────────
    lam = 1.0 / TARGET_BLOCK_TIME
    cdf = 1.0 - np.exp(-lam * df["inter_time_s"].values)
    df["p_value"]  = 2.0 * np.minimum(cdf, 1.0 - cdf)
    df["p_anomaly"] = df["p_value"] < p_threshold

    # ── Z-score on log-transformed times ────────────────────────────────
    # log of Exp(λ) follows a Gumbel; empirically well-approximated by Normal
    log_t  = np.log(df["inter_time_s"].clip(lower=1.0))
    mu_log = np.mean(log_t)
    sd_log = np.std(log_t)
    df["z_score"]  = (log_t - mu_log) / (sd_log + 1e-9)
    df["z_anomaly"] = df["z_score"].abs() > z_threshold

    df["anomaly"] = df["z_anomaly"] | df["p_anomaly"]
    df["date"]    = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    # Classify direction
    df["type"] = "normal"
    df.loc[df["anomaly"] & (df["inter_time_s"] < TARGET_BLOCK_TIME), "type"] = "fast"
    df.loc[df["anomaly"] & (df["inter_time_s"] > TARGET_BLOCK_TIME), "type"] = "slow"

    return df


def classification_report_dict(df: pd.DataFrame, true_col: str, pred_col: str) -> dict:
    """Return precision, recall, F1 treating both columns as boolean."""
    y_true = df[true_col].values.astype(int)
    y_pred = df[pred_col].values.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


# ── Render ──────────────────────────────────────────────────────────────────

def render():
    st.markdown("""
    <div style='font-size:.78rem; color:#64748b; margin-bottom:1rem;'>
    Statistical anomaly detection on block inter-arrival times.
    Expected baseline: <strong style='color:#94a3b8;'>Exp(λ = 1/600 s)</strong>.
    Deviations may indicate mining pool bursts, network partitions, or hash rate surges.
    </div>""", unsafe_allow_html=True)

    # ── Controls ─────────────────────────────────────────────────────────
    ca, cb, cc, cd = st.columns([2, 2, 2, 1])
    with ca:
        n_blocks = st.slider("Blocks to analyse", 30, 200, 80, key="m7_n")
    with cb:
        z_thresh = st.slider("Z-score threshold", 1.5, 4.0, 2.5, step=0.1, key="m7_z")
    with cc:
        p_thresh = st.slider("P-value threshold α", 0.01, 0.20, 0.05, step=0.01, key="m7_p")
    with cd:
        detect_btn = st.button("Detect →", key="m7_go")

    if detect_btn or "m7_df" not in st.session_state:
        with st.spinner("Fetching blocks & running detector…"):
            try:
                blocks = get_recent_blocks(n=n_blocks)
                is_mock = False
            except Exception as exc:
                st.warning(f"API unavailable — demo data. ({exc})")
                blocks  = mock_recent_blocks(n=n_blocks)
                is_mock = True

            df = compute_anomalies(blocks, z_thresh, p_thresh)
            st.session_state.update(m7_df=df, m7_blocks=blocks, m7_mock=is_mock)

    df      = st.session_state["m7_df"]
    is_mock = st.session_state["m7_mock"]

    if df.empty:
        st.error("Not enough blocks to compute inter-arrival times.")
        return

    if is_mock:
        st.caption("⚠️ DEMO DATA — synthetic inter-arrival times.")

    n_anomalies = int(df["anomaly"].sum())
    n_fast      = int((df["type"] == "fast").sum())
    n_slow      = int((df["type"] == "slow").sum())
    pct_anom    = n_anomalies / len(df) * 100
    mean_t      = df["inter_time_s"].mean()
    fitted_lam  = 1.0 / mean_t if mean_t > 0 else 1/600

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Blocks Analysed", str(len(df)),          "dim"),
        ("Anomalies Found", str(n_anomalies),       ""),
        ("Anomaly Rate",    f"{pct_anom:.1f}%",     "blue"),
        ("Fast Blocks",     str(n_fast),            "green"),
        ("Slow Blocks",     str(n_slow),            ""),
    ]
    for col, (label, val, cls) in zip([c1,c2,c3,c4,c5], kpis):
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {cls}">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart row 1 ───────────────────────────────────────────────────────
    l1, r1 = st.columns(2, gap="large")

    with l1:
        st.markdown('<div class="panel-title">Inter-Arrival Time Timeline</div>',
                    unsafe_allow_html=True)
        normal_df  = df[~df["anomaly"]]
        fast_df    = df[df["type"] == "fast"]
        slow_df    = df[df["type"] == "slow"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=normal_df["date"], y=normal_df["inter_time_s"],
            mode="markers", name="Normal",
            marker=dict(color="#3b82f6", size=5, opacity=0.7),
            hovertemplate="%{x|%H:%M}<br>%{y:.0f} s<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=fast_df["date"], y=fast_df["inter_time_s"],
            mode="markers", name="Fast anomaly",
            marker=dict(color="#22c55e", size=10, symbol="triangle-up",
                        line=dict(color="#fff", width=1)),
            hovertemplate="%{x|%H:%M}<br>%{y:.0f} s<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=slow_df["date"], y=slow_df["inter_time_s"],
            mode="markers", name="Slow anomaly",
            marker=dict(color="#ef4444", size=10, symbol="triangle-down",
                        line=dict(color="#fff", width=1)),
            hovertemplate="%{x|%H:%M}<br>%{y:.0f} s<extra></extra>",
        ))
        fig.add_hline(y=600, line_dash="dot", line_color="#f7931a",
                      annotation_text="600 s target",
                      annotation_font=dict(color="#f7931a", size=9))
        fig.update_layout(**PL, height=300, yaxis_title="inter-arrival time (s)",
                          legend=dict(orientation="h", y=1.08, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    with r1:
        st.markdown('<div class="panel-title">Distribution vs Exp(λ = 1/600)</div>',
                    unsafe_allow_html=True)
        t_vals = df["inter_time_s"].values
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=t_vals, nbinsx=20, name="Observed",
            marker_color="#3b82f6", opacity=0.75,
            hovertemplate="%{x:.0f} s: %{y} blocks<extra></extra>",
        ))
        x_th  = np.linspace(0, max(t_vals)*1.2, 300)
        bw    = (max(t_vals) - min(t_vals)) / 20
        lam   = 1.0 / TARGET_BLOCK_TIME
        pdf   = len(t_vals) * bw * lam * np.exp(-lam * x_th)
        # fitted lambda
        lam_f = 1.0 / mean_t
        pdf_f = len(t_vals) * bw * lam_f * np.exp(-lam_f * x_th)

        fig2.add_trace(go.Scatter(
            x=x_th, y=pdf, mode="lines", name="Exp(1/600) theory",
            line=dict(color="#f7931a", width=2, dash="dash"),
        ))
        fig2.add_trace(go.Scatter(
            x=x_th, y=pdf_f, mode="lines", name=f"Exp(1/{mean_t:.0f}) fitted",
            line=dict(color="#a855f7", width=2, dash="dot"),
        ))
        fig2.update_layout(**PL, height=300, xaxis_title="seconds",
                           yaxis_title="count",
                           legend=dict(orientation="h", y=1.08, font=dict(size=10)))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Chart row 2 ───────────────────────────────────────────────────────
    l2, r2 = st.columns(2, gap="large")

    with l2:
        st.markdown('<div class="panel-title">Z-Score per Block</div>',
                    unsafe_allow_html=True)
        colors_z = ["#ef4444" if abs(z) > z_thresh else "#3b82f6"
                    for z in df["z_score"]]
        fig3 = go.Figure(go.Bar(
            x=list(range(len(df))), y=df["z_score"].tolist(),
            marker_color=colors_z, opacity=0.85,
            hovertemplate="block %{x}<br>z=%{y:.2f}<extra></extra>",
        ))
        fig3.add_hline(y= z_thresh, line_dash="dot", line_color="#ef4444",
                       annotation_text=f"+{z_thresh}", annotation_font=dict(size=9))
        fig3.add_hline(y=-z_thresh, line_dash="dot", line_color="#ef4444",
                       annotation_text=f"−{z_thresh}", annotation_font=dict(size=9))
        fig3.update_layout(**PL, height=260, xaxis_title="block index",
                           yaxis_title="z-score (log inter-arrival)")
        st.plotly_chart(fig3, use_container_width=True)

    with r2:
        st.markdown('<div class="panel-title">P-value per Block</div>',
                    unsafe_allow_html=True)
        colors_p = ["#ef4444" if p < p_thresh else "#3b82f6"
                    for p in df["p_value"]]
        fig4 = go.Figure(go.Bar(
            x=list(range(len(df))), y=df["p_value"].tolist(),
            marker_color=colors_p, opacity=0.85,
            hovertemplate="block %{x}<br>p=%{y:.4f}<extra></extra>",
        ))
        fig4.add_hline(y=p_thresh, line_dash="dot", line_color="#f7931a",
                       annotation_text=f"α={p_thresh}",
                       annotation_font=dict(color="#f7931a", size=9))
        fig4.update_layout(**PL, height=260, xaxis_title="block index",
                           yaxis_title="two-sided p-value")
        fig4.update_yaxes(range=[0, 0.3])
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Anomaly table ─────────────────────────────────────────────────────
    if n_anomalies > 0:
        st.markdown('<div class="panel-title">Flagged Blocks</div>',
                    unsafe_allow_html=True)
        anom_df = df[df["anomaly"]][
            ["height", "date", "inter_time_s", "z_score", "p_value", "type"]
        ].copy()
        anom_df["inter_time_s"] = anom_df["inter_time_s"].round(1)
        anom_df["z_score"]      = anom_df["z_score"].round(3)
        anom_df["p_value"]      = anom_df["p_value"].map(lambda x: f"{x:.4f}")
        anom_df["date"]         = anom_df["date"].dt.strftime("%Y-%m-%d %H:%M UTC")
        st.dataframe(anom_df.rename(columns={
            "inter_time_s": "time (s)", "type": "direction"
        }), use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── M4 vs M7 comparison ───────────────────────────────────────────────
    st.markdown('<div class="panel-title">Comparison: M4 (Fee Estimator) vs M7 (Anomaly Detector)</div>',
                unsafe_allow_html=True)

    # Cross-evaluate: use p_anomaly and z_anomaly as two "models" on the same data
    if n_anomalies > 0:
        cross = classification_report_dict(df, "z_anomaly", "p_anomaly")
        st.markdown(f"""
        <div style='font-family:"IBM Plex Mono",monospace; font-size:.73rem;
                    line-height:2.2; color:#64748b;'>
        <span style='color:#94a3b8; font-weight:700;'>Agreement between Z-score and p-value detectors:</span><br>
        Precision: <span style='color:#3b82f6;'>{cross['precision']:.3f}</span> &nbsp;
        Recall: <span style='color:#22c55e;'>{cross['recall']:.3f}</span> &nbsp;
        F1: <span style='color:#f7931a;'>{cross['f1']:.3f}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
| Dimension          | M4 · Fee Estimator                      | M7 · Anomaly Detector                  |
|---|---|---|
| **Problem**        | Predict sat/vByte for next block        | Is this block's timing abnormal?       |
| **Model**          | Gradient Boosting Regressor             | Exponential distribution baseline      |
| **Training data**  | Block-level feature matrix (6 features) | Inter-arrival times only               |
| **Supervision**    | Supervised (known fee targets)          | Unsupervised (no labelled anomalies)   |
| **Output**         | Continuous fee value                    | Binary anomaly flag + direction        |
| **Primary metric** | MAE (sat/vByte), RMSE, MAPE             | P-value, Z-score, F1 between detectors |
| **Key assumption** | Fee predictable from block metadata     | Block times ~ Exp(1/600)               |
    """)

    with st.expander("📐  Statistical background"):
        st.markdown(r"""
**Why exponential?**  
Mining is a memoryless Bernoulli trial process. Each hash attempt succeeds
independently with probability $p = \text{target} / 2^{256}$. With a
large number of attempts per second, the inter-arrival time converges to:

$$T \sim \text{Exp}\!\left(\lambda = \frac{1}{600}\right)$$

**Two-sided p-value:**
$$p = 2 \times \min\!\bigl(F(t), 1 - F(t)\bigr), \quad F(t) = 1 - e^{-\lambda t}$$

A small $p$ means the observed time is in either tail of the distribution
— either blocks are arriving much faster (mining pool burst, increased hash
rate) or much slower (network partition, hash rate drop).

**Z-score on log times:**
Since $\log T$ is approximately Gumbel-distributed (and close to Normal for
samples > 30), the Z-score on $\log t$ gives a complementary signal that is
less sensitive to outlier block times stretching the scale.
        """)
