"""
M4 – AI Fee Estimator  (Gradient Boosting Regressor)
=====================================================
Predicts optimal transaction fee (sat/vByte) from block-level features.

Model choice justification
--------------------------
Gradient Boosting was chosen over linear regression, LSTM or Prophet because:
  • Fees spike non-linearly when blocks are full → tree-based models handle
    this naturally without manual feature transforms.
  • Robust to outliers (extreme fee events during congestion).
  • No feature scaling needed.
  • Feature importance is interpretable and pedagogically useful.

Features
--------
  hour        – UTC hour of block (captures intra-day demand peaks)
  day_of_week – weekday (0=Mon) — weekly demand cycles
  tx_count    – number of confirmed transactions
  size_mb     – block size in MB
  fullness    – size_mb / 1.75 (SegWit soft limit)
  lag_fee     – previous block's median fee (momentum / autocorrelation)

Target: median fee rate per block (sat/vByte)

Evaluation metrics
------------------
  MAE  – Mean Absolute Error  (primary; same unit as target)
  RMSE – Root Mean Squared Error (penalises large misses)
  MAPE – Mean Absolute Percentage Error
  R²   – Coefficient of determination

Reference: Nakamoto (2008) — §6 on fee incentives.
"""

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from api.blockchain_client import (
    get_blocks_with_fees, get_mempool_fees, get_tip_height,
    mock_blocks_with_fees, mock_mempool_fees,
)

PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#94a3b8", size=11),
    margin=dict(t=36, b=36, l=8, r=8),
    xaxis=dict(gridcolor="#1e2535", linecolor="#1e2535", zeroline=False),
    yaxis=dict(gridcolor="#1e2535", linecolor="#1e2535", zeroline=False),
)

FEATURES = ["hour", "day_of_week", "tx_count", "size_mb", "fullness", "lag_fee"]
TARGET   = "median_fee"

# Fee priority tiers (sat/vByte thresholds)
TIERS = [("LOW", 0, 10, "#22c55e"), ("MEDIUM", 10, 40, "#eab308"),
         ("HIGH", 40, 100, "#f97316"), ("PRIORITY", 100, 9999, "#ef4444")]


# ── Data engineering ────────────────────────────────────────────────────────

def to_df(blocks: list[dict]) -> pd.DataFrame:
    rows = []
    for b in blocks:
        extras = b.get("extras", {})
        fee = extras.get("medianFee") or extras.get("avgFeeRate")
        if fee is None:
            continue
        ts   = b.get("timestamp", 0)
        dt   = datetime.fromtimestamp(ts, tz=timezone.utc)
        smb  = b.get("size", 1_000_000) / 1_000_000
        rows.append({
            "height": b.get("height", 0), "timestamp": ts,
            "hour": dt.hour, "day_of_week": dt.weekday(),
            "tx_count": b.get("tx_count", 2000),
            "size_mb": smb, "fullness": min(smb / 1.75, 1.0),
            "median_fee": float(fee),
            "fee_range": extras.get("feeRange", []),
        })
    df = pd.DataFrame(rows).sort_values("height").reset_index(drop=True)
    df["lag_fee"]   = df["median_fee"].shift(1)
    df["lag_fee2"]  = df["median_fee"].shift(2)   # second-order lag
    df["roll_mean"] = df["median_fee"].rolling(5, min_periods=1).mean()
    return df.dropna(subset=["lag_fee"])


def fee_tier(sat_vb: float) -> tuple[str, str]:
    """Return (tier_name, color) for a given fee rate."""
    for name, lo, hi, color in TIERS:
        if lo <= sat_vb < hi:
            return name, color
    return "PRIORITY", "#ef4444"


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ── Model training ──────────────────────────────────────────────────────────

def train(df: pd.DataFrame):
    feats = FEATURES + ["lag_fee2", "roll_mean"]
    df = df.dropna(subset=feats + [TARGET]).reset_index(drop=True)
    X, y  = df[feats], df[TARGET]

    # ── Time-series cross-validation (3 folds, no shuffle) ──
    tscv = TimeSeriesSplit(n_splits=3)
    cv_maes = []
    for tr_idx, te_idx in tscv.split(X):
        m = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08,
                                       max_depth=4, subsample=0.8, random_state=42)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = m.predict(X.iloc[te_idx])
        cv_maes.append(mean_absolute_error(y.iloc[te_idx], preds))

    # ── Final model on 80 / 20 temporal split ──
    split = int(len(X) * 0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte  = y.iloc[:split], y.iloc[split:]

    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08,
                                       max_depth=4, subsample=0.8, random_state=42)
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)

    resid = yte.values - yp
    metrics = dict(
        mae    = mean_absolute_error(yte, yp),
        rmse   = math.sqrt(mean_squared_error(yte, yp)),
        mape   = mape(yte.values, yp),
        r2     = r2_score(yte, yp),
        cv_mae = float(np.mean(cv_maes)),
        cv_std = float(np.std(cv_maes)),
        n_train=len(Xtr), n_test=len(Xte),
    )
    test_df = Xte[FEATURES].copy()
    test_df["actual"], test_df["predicted"], test_df["residual"] = yte.values, yp, resid

    return model, metrics, test_df, feats


# ── Render ──────────────────────────────────────────────────────────────────

def render():
    c_sl, c_btn = st.columns([4, 1])
    with c_sl:
        n_blocks = st.slider("Training blocks", 40, 250, 100, key="m4_n",
                             label_visibility="collapsed")
    with c_btn:
        train_btn = st.button("Train →", key="m4_go")

    if train_btn or "m4_model" not in st.session_state:
        with st.spinner("Fetching block data & training…"):
            try:
                height = get_tip_height()
                raw    = get_blocks_with_fees(height, count=n_blocks)
                mock   = False
            except Exception as exc:
                st.warning(f"API unavailable — synthetic data. ({exc})")
                raw  = mock_blocks_with_fees(count=n_blocks)
                mock = True
            try:
                live = get_mempool_fees()
            except Exception:
                live = mock_mempool_fees(); mock = True

            df = to_df(raw)
            if len(df) < 20:
                st.error("Not enough data — increase block count.")
                return
            model, metrics, test_df, feats = train(df)
            st.session_state.update(
                m4_model=model, m4_metrics=metrics, m4_test=test_df,
                m4_df=df, m4_live=live, m4_mock=mock, m4_feats=feats)

    if "m4_model" not in st.session_state:
        return

    model   = st.session_state["m4_model"]
    metrics = st.session_state["m4_metrics"]
    test_df = st.session_state["m4_test"]
    df      = st.session_state["m4_df"]
    live    = st.session_state["m4_live"]
    mock    = st.session_state["m4_mock"]
    feats   = st.session_state["m4_feats"]

    if mock:
        st.caption("⚠️  DEMO DATA — synthetic training set.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────
    cols = st.columns(6)
    kpis = [
        ("MAE",        f"{metrics['mae']:.2f} sat/vB",   ""),
        ("RMSE",       f"{metrics['rmse']:.2f} sat/vB",  ""),
        ("MAPE",       f"{metrics['mape']:.1f}%",         "blue"),
        ("R²",         f"{metrics['r2']:.3f}",            "blue"),
        ("CV-MAE",     f"{metrics['cv_mae']:.2f} ± {metrics['cv_std']:.2f}", "dim"),
        ("Test rows",  str(metrics["n_test"]),             "dim"),
    ]
    for col, (label, val, cls) in zip(cols, kpis):
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {cls}">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Charts row 1 ─────────────────────────────────────────────────────
    l1, r1 = st.columns(2, gap="large")

    with l1:
        st.markdown('<div class="panel-title">Actual vs Predicted (test set)</div>',
                    unsafe_allow_html=True)
        mx = max(test_df["actual"].max(), test_df["predicted"].max()) * 1.05
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_df["actual"], y=test_df["predicted"], mode="markers",
            name="Block", marker=dict(color="#3b82f6", size=7, opacity=0.7),
            hovertemplate="actual %{x:.1f}<br>pred %{y:.1f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[0, mx], y=[0, mx], mode="lines", name="Perfect",
            line=dict(color="#22c55e", dash="dash", width=1.5),
        ))
        fig.update_layout(**PL, height=280, xaxis_title="actual (sat/vB)",
                          yaxis_title="predicted (sat/vB)",
                          legend=dict(orientation="h", y=1.08, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    with r1:
        st.markdown('<div class="panel-title">Residual Distribution</div>',
                    unsafe_allow_html=True)
        resid = test_df["residual"].values
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=resid, nbinsx=20, name="Residuals",
            marker_color="#a855f7", opacity=0.8,
            hovertemplate="error %{x:.1f} sat/vB<extra></extra>",
        ))
        # Normal overlay
        mu, sigma = float(np.mean(resid)), float(np.std(resid))
        x_n = np.linspace(resid.min()*1.2, resid.max()*1.2, 200)
        bw  = (resid.max() - resid.min()) / 20
        pdf = len(resid) * bw * (1/(sigma*(2*math.pi)**0.5)) * np.exp(-0.5*((x_n-mu)/sigma)**2)
        fig2.add_trace(go.Scatter(
            x=x_n, y=pdf, mode="lines", name=f"N({mu:.1f}, {sigma:.1f}²)",
            line=dict(color="#f7931a", width=2, dash="dash"),
        ))
        fig2.add_vline(x=0, line_dash="dot", line_color="#22c55e",
                       annotation_text="zero error", annotation_font=dict(size=9))
        fig2.update_layout(**PL, height=280, xaxis_title="prediction error (sat/vB)",
                           yaxis_title="count",
                           legend=dict(orientation="h", y=1.08, font=dict(size=10)))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts row 2 ─────────────────────────────────────────────────────
    l2, r2 = st.columns(2, gap="large")

    with l2:
        st.markdown('<div class="panel-title">Feature Importance</div>',
                    unsafe_allow_html=True)
        fi = pd.DataFrame({"Feature": feats,
                           "Importance": model.feature_importances_}).sort_values("Importance")
        colors = ["#3b82f6"] * (len(fi) - 1) + ["#f7931a"]
        fig3 = go.Figure(go.Bar(
            x=fi["Importance"], y=fi["Feature"], orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ))
        fig3.update_layout(**PL, height=280, xaxis_title="importance score")
        st.plotly_chart(fig3, use_container_width=True)

    with r2:
        st.markdown('<div class="panel-title">Predicted vs Block Index (test set)</div>',
                    unsafe_allow_html=True)
        idx = list(range(len(test_df)))
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=idx, y=test_df["actual"].tolist(), mode="lines+markers",
            name="Actual", line=dict(color="#22c55e", width=2),
            marker=dict(size=5),
            hovertemplate="block %{x}<br>%{y:.1f} sat/vB<extra></extra>",
        ))
        fig4.add_trace(go.Scatter(
            x=idx, y=test_df["predicted"].tolist(), mode="lines",
            name="Predicted", line=dict(color="#f7931a", width=2, dash="dot"),
            hovertemplate="block %{x}<br>%{y:.1f} sat/vB<extra></extra>",
        ))
        fig4.update_layout(**PL, height=280, xaxis_title="test block index",
                           yaxis_title="fee (sat/vB)",
                           legend=dict(orientation="h", y=1.08, font=dict(size=10)))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Live prediction widget ────────────────────────────────────────────
    st.markdown('<div class="panel-title">Live Fee Prediction</div>',
                unsafe_allow_html=True)

    now = datetime.now(tz=timezone.utc)
    dow_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}

    ca, cb, cc = st.columns(3)
    with ca:
        pred_hour = st.slider("Hour (UTC)", 0, 23, now.hour, key="m4_hour")
        pred_dow  = st.selectbox("Day", list(dow_map.keys()),
                                 index=now.weekday(), key="m4_dow")
    with cb:
        pred_tx   = st.slider("Expected tx count", 500, 5000, 2500, key="m4_tx")
        pred_full = st.slider("Block fullness %", 0, 100, 85, key="m4_full") / 100
    with cc:
        pred_lag  = st.slider("Prev block fee (sat/vB)", 1, 200, 40, key="m4_lag")
        pred_lag2 = st.slider("2nd-prev block fee (sat/vB)", 1, 200, 38, key="m4_lag2")

    pred_size = pred_tx * 450 / 1_000_000
    roll_m    = (pred_lag + pred_lag2) / 2  # simplified rolling mean
    X_new = pd.DataFrame([dict(
        hour=pred_hour, day_of_week=dow_map[pred_dow],
        tx_count=pred_tx, size_mb=pred_size, fullness=pred_full,
        lag_fee=float(pred_lag), lag_fee2=float(pred_lag2), roll_mean=roll_m,
    )])
    pred_fee = max(1.0, float(model.predict(X_new)[0]))
    tier_name, tier_color = fee_tier(pred_fee)

    # Comparison cards: model + live tiers
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    live_cards = [
        ("🤖 Model", f"{pred_fee:.1f}",                        tier_color, "sat/vByte"),
        ("Fastest",  str(live.get("fastestFee",  "?")),         "#ef4444",  "sat/vByte"),
        ("½ Hour",   str(live.get("halfHourFee", "?")),         "#f97316",  "sat/vByte"),
        ("1 Hour",   str(live.get("hourFee",     "?")),         "#eab308",  "sat/vByte"),
        ("Economy",  str(live.get("economyFee",  "?")),         "#22c55e",  "sat/vByte"),
    ]
    for col, (label, val, color, unit) in zip([fc1,fc2,fc3,fc4,fc5], live_cards):
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color};">{val}</div>
            <div class="kpi-sub">{unit}</div>
        </div>""", unsafe_allow_html=True)

    # Priority tier indicator
    st.markdown(f"""
    <div style='margin-top:.75rem; padding:.6rem 1rem; background:#111520;
                border:1px solid {tier_color}; border-radius:8px; display:inline-block;'>
        <span style='font-size:.70rem; font-weight:700; letter-spacing:.10em;
                     text-transform:uppercase; color:#64748b;'>Predicted priority tier</span>
        <span style='font-family:"IBM Plex Mono",monospace; font-size:1.1rem;
                     font-weight:700; color:{tier_color}; margin-left:1rem;'>
            {tier_name}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Fee tier distribution over training data ──────────────────────────
    st.markdown('<div class="panel-title">Fee Tier Distribution (training data)</div>',
                unsafe_allow_html=True)

    la, ra = st.columns(2, gap="large")
    with la:
        df_plot = df.copy()
        df_plot["Date"] = pd.to_datetime(df_plot["timestamp"], unit="s", utc=True)
        fig5 = go.Figure()
        # Background tier bands
        for tname, lo, hi, tc in TIERS:
            fig5.add_hrect(y0=lo, y1=min(hi, df_plot["median_fee"].max()*1.1),
                           fillcolor=tc, opacity=0.06, line_width=0,
                           annotation_text=tname, annotation_position="right",
                           annotation=dict(font=dict(color=tc, size=9)))
        fig5.add_trace(go.Scatter(
            x=df_plot["Date"], y=df_plot["median_fee"], mode="lines",
            fill="tozeroy", line=dict(color="#a855f7", width=2),
            fillcolor="rgba(168,85,247,0.08)",
            hovertemplate="%{x|%b %d %H:%M}<br>%{y:.1f} sat/vB<extra></extra>",
        ))
        fig5.update_layout(**PL, height=260, yaxis_title="median fee (sat/vB)")
        st.plotly_chart(fig5, use_container_width=True)

    with ra:
        counts = {tname: int(((df["median_fee"] >= lo) & (df["median_fee"] < hi)).sum())
                  for tname, lo, hi, _ in TIERS}
        fig6 = go.Figure(go.Bar(
            x=list(counts.keys()), y=list(counts.values()),
            marker_color=[t[3] for t in TIERS],
            hovertemplate="%{x}: %{y} blocks<extra></extra>",
        ))
        fig6.update_layout(**PL, height=260, yaxis_title="block count",
                           xaxis_title="fee tier")
        st.plotly_chart(fig6, use_container_width=True)

    with st.expander("ℹ️  Full evaluation notes"):
        st.markdown(f"""
**MAE = {metrics['mae']:.2f} sat/vByte** — average absolute prediction error.  
**RMSE = {metrics['rmse']:.2f}** — heavier penalty on large misses (congestion spikes).  
**MAPE = {metrics['mape']:.1f}%** — scale-independent; useful when fees span a wide range.  
**R² = {metrics['r2']:.3f}** — fraction of variance explained by the model.  
**Cross-validated MAE = {metrics['cv_mae']:.2f} ± {metrics['cv_std']:.2f}** (3-fold TimeSeriesSplit) — gives a more robust estimate than a single train/test split by respecting temporal order.

**Limitations:**  
- The strongest real-world predictor is current mempool depth (pending tx backlog),  
  which requires a live WebSocket to mempool.space and is not yet integrated.  
- The model is trained on block-level data. During sudden demand spikes, the  
  lag features capture momentum but cannot anticipate the spike onset.  
- **Temporal split** (no shuffle) is used throughout to prevent look-ahead bias.

**Feature engineering:**  
Added `lag_fee2` (2-block lag) and `roll_mean` (5-block rolling average) to capture  
short-term fee autocorrelation beyond the single-lag baseline.
        """)
