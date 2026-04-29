"""
M4 – AI Component: Transaction Fee Estimator
==============================================
Approach chosen: Supervised regression model to predict the optimal
transaction fee (sat/vByte) from block and mempool features.

Pipeline:
  1. Fetch historical blocks with fee statistics (Mempool.space API).
  2. Engineer features: hour-of-day, day-of-week, block size, tx count,
     block fullness ratio, rolling mean fee (lag features).
  3. Train a Gradient Boosting Regressor (scikit-learn).
  4. Evaluate with MAE and R² on a held-out test set.
  5. Show current fee predictions vs Mempool.space recommendations.

Why Gradient Boosting?
  • Handles non-linear relationships (fees spike non-linearly at congestion).
  • Robust to outliers (fee spikes during bull markets).
  • No need to scale features.
  • Interpretable via feature importance.

References:
  • Mempool.space API: https://mempool.space/docs/api
  • Nakamoto (2008) — Bitcoin: A Peer-to-Peer Electronic Cash System
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from api.blockchain_client import (
    get_mempool_fees,
    get_tip_height,
    get_blocks_with_fees,
    mock_blocks_with_fees,
    mock_mempool_fees,
)

# ── Feature engineering ─────────────────────────────────────────────────────

def blocks_to_dataframe(blocks: list[dict]) -> pd.DataFrame:
    """
    Extract features and target from a list of Mempool.space block dicts.

    Features:
      hour          — UTC hour of block timestamp (captures intra-day demand)
      day_of_week   — 0=Monday … 6=Sunday (captures weekly demand cycles)
      tx_count      — number of transactions in the block
      size_mb       — block size in megabytes
      fullness      — size_mb / 1.75 (soft limit for SegWit blocks)
      lag_fee       — previous block's median fee (momentum feature)

    Target:
      median_fee    — median fee rate in sat/vByte
    """
    rows = []
    for b in blocks:
        ts = b.get("timestamp", 0)
        extras = b.get("extras", {})
        median_fee = extras.get("medianFee") or extras.get("avgFeeRate")
        if median_fee is None:
            continue
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        size_mb = b.get("size", 1_000_000) / 1_000_000
        tx_count = b.get("tx_count", 2000)
        rows.append({
            "height": b.get("height", 0),
            "timestamp": ts,
            "hour": dt.hour,
            "day_of_week": dt.weekday(),
            "tx_count": tx_count,
            "size_mb": size_mb,
            "fullness": min(size_mb / 1.75, 1.0),   # 1.75 MB ≈ typical max block weight/4
            "median_fee": float(median_fee),
        })

    df = pd.DataFrame(rows).sort_values("height").reset_index(drop=True)
    # Lag feature: previous block median fee
    df["lag_fee"] = df["median_fee"].shift(1)
    df = df.dropna()
    return df


FEATURE_COLS = ["hour", "day_of_week", "tx_count", "size_mb", "fullness", "lag_fee"]
TARGET_COL   = "median_fee"


def train_model(df: pd.DataFrame):
    """Train a Gradient Boosting Regressor and return (model, metrics, test_df)."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # keep temporal order
    )

    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    test_df = X_test.copy()
    test_df["actual"] = y_test.values
    test_df["predicted"] = y_pred

    return model, metrics, test_df


# ── Render ──────────────────────────────────────────────────────────────────

def render() -> None:
    st.header("🤖 M4 – AI Fee Estimator")
    st.markdown(
        "A **Gradient Boosting** regression model trained on recent block data "
        "to predict the optimal transaction fee rate (sat/vByte)."
    )

    with st.expander("ℹ️ Model design & data pipeline", expanded=False):
        st.markdown(
            """
**Why fee estimation?**  
The Bitcoin mempool operates as a fee-priority queue. Miners select
transactions that maximise their revenue (sat/vByte × vBytes). Predicting
the clearing fee rate lets users decide whether to pay a premium for fast
confirmation or wait for a cheaper slot.

**Model: Gradient Boosting Regressor**  
| Reason | Detail |
|---|---|
| Non-linear relationships | Fees spike non-linearly at full blocks |
| Outlier robustness | Handles rare fee spikes during congestion events |
| No feature scaling needed | Tree-based model |
| Feature importance | Interpretable output |

**Features used:**  
`hour`, `day_of_week`, `tx_count`, `size_mb`, `fullness`, `lag_fee`

**Target:** median fee rate (sat/vByte) per block  
**Split:** 80% train / 20% test (temporal, no shuffle)  
**Evaluation:** MAE (primary) + R²
            """
        )

    # ── Data loading ──────────────────────────────────────────────────────
    col_n, col_btn = st.columns([3, 1])
    with col_n:
        n_blocks = st.slider(
            "Training blocks (recent history)",
            min_value=30, max_value=200, value=80, key="m4_n_blocks"
        )
    with col_btn:
        train_btn = st.button("🚀 Train model", key="m4_train")

    if "m4_model" not in st.session_state:
        st.session_state["m4_model"] = None

    if train_btn or st.session_state["m4_model"] is None:
        with st.spinner("Fetching block data & training model…"):
            try:
                height = get_tip_height()
                raw_blocks = get_blocks_with_fees(height, count=n_blocks)
                is_mock = False
            except Exception as exc:
                st.warning(f"Live API unavailable ({exc}). Using synthetic training data.")
                raw_blocks = mock_blocks_with_fees(count=n_blocks)
                is_mock = True

            try:
                live_fees = get_mempool_fees()
            except Exception:
                live_fees = mock_mempool_fees()
                is_mock = True

            df = blocks_to_dataframe(raw_blocks)
            if len(df) < 15:
                st.error("Not enough data points to train. Try increasing the number of blocks.")
                return

            model, metrics, test_df = train_model(df)

            st.session_state["m4_model"] = model
            st.session_state["m4_metrics"] = metrics
            st.session_state["m4_test_df"] = test_df
            st.session_state["m4_df"] = df
            st.session_state["m4_live_fees"] = live_fees
            st.session_state["m4_mock"] = is_mock

    if st.session_state["m4_model"] is None:
        return

    model = st.session_state["m4_model"]
    metrics: dict = st.session_state["m4_metrics"]
    test_df: pd.DataFrame = st.session_state["m4_test_df"]
    df: pd.DataFrame = st.session_state["m4_df"]
    live_fees: dict = st.session_state["m4_live_fees"]
    is_mock: bool = st.session_state["m4_mock"]

    if is_mock:
        st.caption("⚠️ DEMO DATA — model trained on synthetic fee data.")

    st.divider()

    # ── Model performance ─────────────────────────────────────────────────
    st.subheader("📊 Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{metrics['mae']:.2f} sat/vByte",
              help="Mean Absolute Error — average prediction error in sat/vByte")
    c2.metric("R²", f"{metrics['r2']:.3f}",
              help="Coefficient of determination (1.0 = perfect)")
    c3.metric("Training samples", metrics["n_train"])
    c4.metric("Test samples", metrics["n_test"])

    # Actual vs Predicted scatter
    fig_pred = px.scatter(
        test_df, x="actual", y="predicted",
        title="Actual vs Predicted Fee Rate (test set)",
        labels={"actual": "Actual (sat/vByte)", "predicted": "Predicted (sat/vByte)"},
        color_discrete_sequence=["#3b82f6"],
        opacity=0.7,
    )
    max_val = max(test_df["actual"].max(), test_df["predicted"].max()) * 1.05
    fig_pred.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", name="Perfect prediction",
        line=dict(color="#22c55e", dash="dash"),
    ))
    fig_pred.update_layout(height=380)
    st.plotly_chart(fig_pred, use_container_width=True)

    st.divider()

    # ── Feature importance ────────────────────────────────────────────────
    st.subheader("🔎 Feature Importance")
    fi = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig_fi = px.bar(
        fi, x="Importance", y="Feature", orientation="h",
        title="Gradient Boosting Feature Importances",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig_fi.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

    # ── Live fee prediction ───────────────────────────────────────────────
    st.subheader("💸 Live Fee Prediction")
    st.markdown("Adjust the current network state to get a fee prediction.")

    now_utc = datetime.now(tz=timezone.utc)

    col_a, col_b = st.columns(2)
    with col_a:
        pred_hour = st.slider("Hour of day (UTC)", 0, 23, now_utc.hour, key="m4_hour")
        pred_dow  = st.selectbox("Day of week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                                 index=now_utc.weekday(), key="m4_dow")
        dow_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
    with col_b:
        pred_tx   = st.slider("Expected tx count", 500, 5000, 2500, key="m4_tx")
        pred_full = st.slider("Block fullness %", 0, 100, 85, key="m4_full") / 100
        pred_lag  = st.slider("Previous block fee (sat/vByte)", 1, 200, 40, key="m4_lag")

    pred_size = pred_tx * 450 / 1_000_000   # approximate size from tx count

    X_new = pd.DataFrame([{
        "hour": pred_hour,
        "day_of_week": dow_map[pred_dow],
        "tx_count": pred_tx,
        "size_mb": pred_size,
        "fullness": pred_full,
        "lag_fee": float(pred_lag),
    }])
    predicted_fee = float(model.predict(X_new)[0])

    col_pred, col_api = st.columns(2)
    with col_pred:
        st.markdown("#### 🤖 Model Prediction")
        st.metric("Recommended fee", f"{max(1.0, predicted_fee):.1f} sat/vByte")
    with col_api:
        st.markdown("#### 📡 Mempool.space Live")
        st.metric("Fastest fee",    f"{live_fees.get('fastestFee', '?')} sat/vByte")
        st.metric("½-hour fee",     f"{live_fees.get('halfHourFee', '?')} sat/vByte")
        st.metric("Economy fee",    f"{live_fees.get('economyFee', '?')} sat/vByte")

    st.divider()

    # ── Historical fee trend ──────────────────────────────────────────────
    st.subheader("📉 Historical Median Fee Rate (training data)")
    df_plot = df.copy()
    df_plot["Date"] = pd.to_datetime(df_plot["timestamp"], unit="s", utc=True)
    fig_trend = px.line(
        df_plot, x="Date", y="median_fee",
        title="Median Fee Rate per Block",
        labels={"median_fee": "Median fee (sat/vByte)", "Date": ""},
        color_discrete_sequence=["#a855f7"],
    )
    fig_trend.update_layout(height=320)
    st.plotly_chart(fig_trend, use_container_width=True)

    with st.expander("📋 Evaluation notes"):
        st.markdown(
            f"""
**MAE = {metrics['mae']:.2f} sat/vByte** — the model's average absolute prediction
error on the held-out test set. For context, fees typically range from
1–200 sat/vByte, so an MAE of a few sat/vByte is acceptable for
priority-tier estimation.

**R² = {metrics['r2']:.3f}** — explains {'good share of' if metrics['r2'] > 0.5 else 'some of'} the variance
in fee rates. Low R² on fee data is expected: fee markets are driven by
real-time mempool dynamics (sudden demand spikes, whale transactions)
that cannot be predicted from block-level features alone.

**Limitations:**
- Real-time mempool depth is the strongest predictor but requires a
  WebSocket connection (future work).
- Model is retrained on page load — for production, persist the model
  with `joblib.dump`.
- Evaluation uses a temporal split (no shuffle) to avoid look-ahead bias.
            """
        )
