"""
M3 – Difficulty History
========================
Plots the evolution of Bitcoin's mining difficulty over time.

Key concepts:
  • One difficulty adjustment epoch = 2016 blocks ≈ 14 days.
  • The network targets one block every 600 seconds (10 minutes).
  • Adjustment formula (Bitcoin Core):
        new_difficulty = old_difficulty × (actual_time / target_time)
    where target_time = 2016 × 600 = 1 209 600 seconds.
  • To prevent extreme swings, the ratio is clamped to [0.25, 4].
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api.blockchain_client import (
    get_difficulty_history,
    mock_difficulty_history,
)

TARGET_BLOCK_TIME = 600          # seconds
EPOCH_BLOCKS = 2016             # blocks per difficulty epoch
EPOCH_SECONDS = EPOCH_BLOCKS * TARGET_BLOCK_TIME  # 1 209 600 s ≈ 14 days


def _detect_adjustment_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark rows where a significant difficulty adjustment occurred.
    We flag points where difficulty changed by more than 1 % vs the previous point.
    """
    df = df.copy()
    df["pct_change"] = df["Difficulty"].pct_change().abs()
    df["is_adjustment"] = df["pct_change"] > 0.01
    return df


def _compute_block_time_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the ratio actual_block_time / 600 s for each epoch.

    Between two consecutive data-points we know elapsed wall-clock time.
    Difficulty adjusts every 2016 blocks. The adjustment formula implies:
        actual_block_time ≈ target_block_time × (D_old / D_new)
    so:
        ratio = D_old / D_new
    """
    df = df.copy()
    df["ratio"] = df["Difficulty"].shift(1) / df["Difficulty"]
    # Clamp to the protocol limits (×0.25 … ×4 in actual adjustment)
    df["ratio"] = df["ratio"].clip(0.1, 5.0)
    return df


# ── Render ──────────────────────────────────────────────────────────────────

def render() -> None:
    st.header("📈 M3 – Difficulty History")
    st.markdown(
        "Bitcoin's mining difficulty adjusts every **2 016 blocks** "
        "(≈ 14 days) so that the average inter-block time stays close to "
        "**600 seconds**. Here we visualise the historical trend and each "
        "adjustment event."
    )

    col_sl, col_btn = st.columns([3, 1])
    with col_sl:
        n_points = st.slider(
            "Data points (sampled over the past year)",
            min_value=20, max_value=365, value=120, key="m3_n"
        )
    with col_btn:
        load = st.button("📊 Load chart", key="m3_load")

    if "m3_df" not in st.session_state:
        st.session_state["m3_df"] = None
        st.session_state["m3_mock"] = False

    if load or st.session_state["m3_df"] is None:
        with st.spinner("Fetching difficulty history…"):
            try:
                values = get_difficulty_history(n_points)
                st.session_state["m3_mock"] = False
            except Exception as exc:
                st.warning(f"Live API unavailable ({exc}). Showing demo data.")
                values = mock_difficulty_history(n_points)
                st.session_state["m3_mock"] = True

            df = pd.DataFrame(values)
            df["Date"] = pd.to_datetime(df["x"], unit="s", utc=True)
            df = df.rename(columns={"y": "Difficulty"})
            df = df[["Date", "Difficulty"]].dropna().sort_values("Date")
            df = _detect_adjustment_events(df)
            df = _compute_block_time_ratio(df)
            st.session_state["m3_df"] = df

    df: pd.DataFrame = st.session_state["m3_df"]
    is_mock: bool = st.session_state["m3_mock"]

    if is_mock:
        st.caption("⚠️ DEMO DATA — synthetic difficulty trend.")

    # ── KPI row ───────────────────────────────────────────────────────────
    latest_d = df["Difficulty"].iloc[-1]
    first_d = df["Difficulty"].iloc[0]
    pct_change = (latest_d / first_d - 1) * 100
    n_adj = int(df["is_adjustment"].sum())
    avg_ratio = df["ratio"].dropna().mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current difficulty", f"{latest_d:,.0f}")
    c2.metric("Change over period", f"{pct_change:+.1f}%")
    c3.metric("Adjustment events detected", n_adj)
    c4.metric("Avg block-time ratio", f"{avg_ratio:.3f}×")

    st.divider()

    # ── Chart 1: Difficulty over time ─────────────────────────────────────
    st.subheader("Difficulty Over Time")

    adj_df = df[df["is_adjustment"]]
    fig_diff = px.line(
        df, x="Date", y="Difficulty",
        title="Bitcoin Mining Difficulty (log scale)",
        labels={"Difficulty": "Difficulty", "Date": ""},
        color_discrete_sequence=["#3b82f6"],
    )
    fig_diff.update_yaxes(type="log", title="Difficulty (log₁₀)")
    # Mark adjustment events
    fig_diff.add_trace(go.Scatter(
        x=adj_df["Date"],
        y=adj_df["Difficulty"],
        mode="markers",
        name="Adjustment event",
        marker=dict(color="#f97316", size=8, symbol="diamond"),
    ))
    fig_diff.update_layout(height=420, legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig_diff, use_container_width=True)

    st.divider()

    # ── Chart 2: Block-time ratio ─────────────────────────────────────────
    st.subheader("Actual Block Time vs Target (600 s)")
    st.markdown(
        "A ratio **> 1** means blocks were arriving **slower** than 600 s "
        "(difficulty will decrease next epoch). "
        "A ratio **< 1** means blocks were arriving **faster** (difficulty will increase)."
    )

    ratio_df = df.dropna(subset=["ratio"])
    fig_ratio = px.bar(
        ratio_df, x="Date", y="ratio",
        title="Block-Time Ratio = D_old / D_new (per sampled epoch)",
        color="ratio",
        color_continuous_scale=[(0, "#22c55e"), (0.5, "#eab308"), (1, "#ef4444")],
        range_color=[0.5, 1.5],
        labels={"ratio": "Ratio", "Date": ""},
    )
    fig_ratio.add_hline(y=1.0, line_dash="dash", line_color="white",
                        annotation_text="Target (ratio = 1)")
    fig_ratio.add_hline(y=0.25, line_dash="dot", line_color="#9ca3af",
                        annotation_text="Protocol floor (×0.25)")
    fig_ratio.add_hline(y=4.0,  line_dash="dot", line_color="#9ca3af",
                        annotation_text="Protocol ceiling (×4)")
    fig_ratio.update_layout(height=350)
    st.plotly_chart(fig_ratio, use_container_width=True)

    st.divider()

    # ── Adjustment formula explainer ──────────────────────────────────────
    st.subheader("🧮 Difficulty Adjustment Formula")
    st.markdown(
        r"""
Every **2 016 blocks** the Bitcoin protocol recalculates difficulty:

$$\text{new\_difficulty} = \text{old\_difficulty} \times \frac{\text{actual time for 2016 blocks}}{\text{target time} = 2016 \times 600\,s}$$

**Protocol safeguard:** the ratio is clamped to **[¼, 4]** to prevent
extreme swings caused by sudden hash rate drops or surges.

**Example:** if 2 016 blocks were mined in 12 days instead of 14 days,  
ratio = 12/14 ≈ 0.857 → new difficulty ≈ old × 0.857 (−14.3 %, harder to mine … wait, 
faster blocks → shorter actual time → ratio < 1 → 
new_difficulty_target is *smaller* → mining is *harder*).
        """
    )

    with st.expander("📋 Raw data table"):
        disp = df[["Date", "Difficulty", "pct_change", "ratio"]].copy()
        disp["pct_change"] = disp["pct_change"].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
        disp["ratio"] = disp["ratio"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
        disp["Difficulty"] = disp["Difficulty"].map(lambda x: f"{x:,.0f}")
        st.dataframe(disp.rename(columns={"pct_change": "Δ difficulty", "ratio": "Block-time ratio"}),
                     use_container_width=True, hide_index=True)
