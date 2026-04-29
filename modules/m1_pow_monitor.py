"""
M1 – Proof of Work Monitor
===========================
Shows live Bitcoin mining data:
  • Current difficulty and its representation as a leading-zero threshold.
  • Distribution of inter-block times for the last N blocks.
  • Estimated current network hash rate.
"""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api.blockchain_client import (
    get_latest_block,
    get_recent_blocks,
    mock_latest_block,
    mock_recent_blocks,
)

# ── Cryptographic helpers ───────────────────────────────────────────────────

def bits_to_target(bits: int) -> int:
    """
    Decode the compact 'bits' field into the 256-bit target integer.

    Format: bits = (exponent << 24) | coefficient
    Target  = coefficient * 2^(8*(exponent - 3))

    A valid block hash (in little-endian byte order) must be ≤ target.
    """
    exponent = bits >> 24
    coefficient = bits & 0x00FF_FFFF
    return coefficient * (2 ** (8 * (exponent - 3)))


def target_to_difficulty(target: int) -> float:
    """
    difficulty = genesis_target / current_target
    genesis_target corresponds to bits = 0x1d00ffff
    """
    genesis_target = 0x00FFFF * (2 ** (8 * (0x1D - 3)))
    return genesis_target / target


def leading_zero_bits(target: int) -> int:
    """Count how many leading zero *bits* the target demands."""
    # The hash must be ≤ target, so we count leading zeros in the 256-bit space.
    return 256 - target.bit_length()


def estimate_hashrate(difficulty: float) -> float:
    """
    Estimated network hash rate in hashes/second.

    From the protocol: on average, 2^32 * difficulty hashes are needed
    to find a valid block at 600-second block time.
    """
    return difficulty * (2 ** 32) / 600


# ── Render ──────────────────────────────────────────────────────────────────

def render() -> None:
    st.header("⛏️ M1 – Proof of Work Monitor")
    st.markdown(
        "Live Bitcoin mining statistics. "
        "Data refreshes automatically every 60 seconds."
    )

    # ── Auto-refresh counter ──────────────────────────────────────────────
    if "m1_last_fetch" not in st.session_state:
        st.session_state["m1_last_fetch"] = 0
        st.session_state["m1_block"] = None
        st.session_state["m1_recent"] = None
        st.session_state["m1_mock"] = False

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        refresh = st.button("🔄 Refresh now", key="m1_refresh")

    elapsed = time.time() - st.session_state["m1_last_fetch"]
    needs_fetch = refresh or elapsed > 60 or st.session_state["m1_block"] is None

    if needs_fetch:
        with st.spinner("Fetching live data from Blockstream…"):
            try:
                block = get_latest_block()
                recent = get_recent_blocks(n=30)
                st.session_state["m1_mock"] = False
            except Exception as exc:
                st.warning(f"Live API unavailable ({exc}). Showing demo data.")
                block = mock_latest_block()
                recent = mock_recent_blocks(n=30)
                st.session_state["m1_mock"] = True

            st.session_state["m1_block"] = block
            st.session_state["m1_recent"] = recent
            st.session_state["m1_last_fetch"] = time.time()

    block: dict = st.session_state["m1_block"]
    recent: list[dict] = st.session_state["m1_recent"]
    is_mock: bool = st.session_state["m1_mock"]

    with col_status:
        src = "⚠️ DEMO DATA" if is_mock else "✅ Live"
        last = datetime.fromtimestamp(
            st.session_state["m1_last_fetch"], tz=timezone.utc
        ).strftime("%H:%M:%S UTC")
        st.caption(f"{src}  |  Last updated: {last}  |  Auto-refresh every 60 s")

    # ── Decode bits ────────────────────────────────────────────────────────
    bits: int = block.get("bits", 0x1703A30C)
    target = bits_to_target(bits)
    difficulty = target_to_difficulty(target)
    leading_zeros = leading_zero_bits(target)
    hashrate = estimate_hashrate(difficulty)

    # ── Top KPI cards ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Block height", f"{block.get('height', 'N/A'):,}")
    c2.metric("Transactions", f"{block.get('tx_count', 'N/A'):,}")
    c3.metric("Nonce", f"{block.get('nonce', 'N/A'):,}")
    c4.metric(
        "Block size",
        f"{block.get('size', 0) / 1_000_000:.2f} MB",
    )

    st.divider()

    # ── Difficulty & Target visualisation ──────────────────────────────────
    st.subheader("🔐 Difficulty & Proof-of-Work Target")

    col_l, col_r = st.columns(2)
    with col_l:
        st.metric("Difficulty", f"{difficulty:,.0f}")
        st.metric("bits field (hex)", hex(bits))
        st.metric("Leading zero bits required", f"{leading_zeros} / 256")
        hr_unit = hashrate / 1e18
        st.metric("Estimated network hash rate", f"{hr_unit:.2f} EH/s")

    with col_r:
        # Visual: 256-bit space as a horizontal bar
        fraction_zeros = leading_zeros / 256
        fig_bar = go.Figure(go.Bar(
            x=[fraction_zeros * 100, (1 - fraction_zeros) * 100],
            y=["256-bit space"],
            orientation="h",
            marker_color=["#ef4444", "#22c55e"],
            text=[f"{leading_zeros} zero bits (invalid zone)", "Valid hash zone"],
            textposition="inside",
        ))
        fig_bar.update_layout(
            title="SHA-256 Hash Space — Required Leading Zeros",
            barmode="stack",
            xaxis_title="% of 256-bit space",
            showlegend=False,
            height=160,
            margin=dict(t=40, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        target_hex = f"{target:064x}"
        st.markdown(
            f"**256-bit target (first 32 chars):**  \n"
            f"`{target_hex[:32]}…`"
        )

    # Explain the bits field
    exp = bits >> 24
    coef = bits & 0x00FF_FFFF
    st.info(
        f"**How bits → target works:** bits = `{hex(bits)}` means  \n"
        f"exponent = `{exp}`, coefficient = `{hex(coef)}`  \n"
        f"→ target = {hex(coef)} × 2^(8×({exp}−3)) = `{target_hex[:20]}…`  \n"
        f"A valid block hash must be **less than or equal to this target**."
    )

    st.divider()

    # ── Inter-block time distribution ─────────────────────────────────────
    st.subheader("⏱️ Inter-Block Time Distribution (last 30 blocks)")
    st.markdown(
        "Block discovery follows a **memoryless Poisson process** — each "
        "attempt is independent, so inter-arrival times follow an "
        "**exponential distribution** with mean ≈ 600 seconds."
    )

    if len(recent) >= 2:
        timestamps = sorted([b["timestamp"] for b in recent], reverse=True)
        inter_times = [timestamps[i] - timestamps[i + 1] for i in range(len(timestamps) - 1)]
        df_times = pd.DataFrame({"Inter-block time (s)": inter_times})

        fig_hist = px.histogram(
            df_times,
            x="Inter-block time (s)",
            nbins=20,
            title="Distribution of Inter-Block Times",
            labels={"Inter-block time (s)": "Seconds between consecutive blocks"},
            color_discrete_sequence=["#3b82f6"],
        )
        # Overlay theoretical exponential PDF
        x_vals = np.linspace(0, max(inter_times) * 1.2, 300)
        lambda_ = 1 / 600
        bin_width = (max(inter_times) - min(inter_times)) / 20
        n_obs = len(inter_times)
        pdf_vals = n_obs * bin_width * lambda_ * np.exp(-lambda_ * x_vals)
        fig_hist.add_trace(go.Scatter(
            x=x_vals, y=pdf_vals,
            mode="lines",
            name="Exp(λ=1/600) theory",
            line=dict(color="#f97316", width=2, dash="dash"),
        ))
        fig_hist.add_vline(x=600, line_dash="dot", line_color="#22c55e",
                           annotation_text="Target 600 s")
        fig_hist.update_layout(height=380)
        st.plotly_chart(fig_hist, use_container_width=True)

        mean_t = np.mean(inter_times)
        median_t = np.median(inter_times)
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean inter-block time", f"{mean_t:.0f} s")
        c2.metric("Median inter-block time", f"{median_t:.0f} s")
        c3.metric("Blocks sampled", len(inter_times))
    else:
        st.info("Not enough blocks fetched for the histogram.")

    st.divider()

    # ── Latest block hash ─────────────────────────────────────────────────
    st.subheader("🔗 Latest Block Hash")
    block_hash = block.get("id") or block.get("hash", "N/A")
    st.code(block_hash, language=None)
    zeros = len(block_hash) - len(block_hash.lstrip("0"))
    st.caption(
        f"Leading hex zeros: **{zeros}** (≈ {zeros * 4} leading zero bits). "
        "Each hex '0' represents exactly 4 zero bits."
    )
