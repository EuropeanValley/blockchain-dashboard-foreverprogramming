"""M1 – Proof of Work Monitor"""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api.blockchain_client import get_latest_block, get_recent_blocks, mock_latest_block, mock_recent_blocks

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#94a3b8", size=11),
    margin=dict(t=36, b=36, l=8, r=8),
    xaxis=dict(gridcolor="#1e2535", linecolor="#1e2535", zeroline=False),
    yaxis=dict(gridcolor="#1e2535", linecolor="#1e2535", zeroline=False),
)

def bits_to_target(bits):
    return (bits & 0x00FF_FFFF) * (2 ** (8 * ((bits >> 24) - 3)))

def target_to_difficulty(target):
    return 0x00FFFF * (2 ** (8 * (0x1D - 3))) / target

def leading_zero_bits(target):
    return 256 - target.bit_length()

def estimate_hashrate(diff):
    return diff * (2 ** 32) / 600

def _fetch():
    try:
        block = get_latest_block()
        recent = get_recent_blocks(n=30)
        return block, recent, False
    except Exception as exc:
        return mock_latest_block(), mock_recent_blocks(n=30), True


def render():
    # ── Auto-refresh every 60 s ──────────────────────────────────────────
    col_ref, col_status = st.columns([1, 5])
    with col_ref:
        do_refresh = st.button("↺  Refresh", key="m1_refresh")

    elapsed = time.time() - st.session_state.get("m1_ts", 0)
    if do_refresh or elapsed > 60 or "m1_block" not in st.session_state:
        with st.spinner(""):
            block, recent, is_mock = _fetch()
        st.session_state.update(m1_block=block, m1_recent=recent,
                                m1_mock=is_mock, m1_ts=time.time())

    block  = st.session_state["m1_block"]
    recent = st.session_state["m1_recent"]
    mock   = st.session_state["m1_mock"]

    with col_status:
        pill = '<span class="pill pill-demo">⚠ demo</span>' if mock else '<span class="pill pill-live">● live</span>'
        last = datetime.fromtimestamp(st.session_state["m1_ts"], tz=timezone.utc).strftime("%H:%M:%S UTC")
        st.markdown(f"{pill} &nbsp; <span style='font-size:.72rem;color:#64748b;'>last updated {last} · auto-refresh 60 s</span>",
                    unsafe_allow_html=True)

    # ── Derived values ───────────────────────────────────────────────────
    bits       = block.get("bits", 0x1703A30C)
    target     = bits_to_target(bits)
    difficulty = target_to_difficulty(target)
    lz_bits    = leading_zero_bits(target)
    hashrate   = estimate_hashrate(difficulty)
    target_hex = f"{target:064x}"

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 1: KPI cards ─────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Block Height",     f"{block.get('height',0):,}",            "dim"),
        (c2, "Difficulty",       f"{difficulty/1e12:.3f} T",               ""),
        (c3, "Hash Rate",        f"{hashrate/1e18:.2f} EH/s",             "blue"),
        (c4, "Leading Zero Bits",f"{lz_bits} / 256",                      "green"),
        (c5, "Transactions",     f"{block.get('tx_count',0):,}",           "dim"),
    ]
    for col, label, value, cls in cards:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {cls}">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 2: Charts side by side ───────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="panel-title">256-bit SHA-256 Target Space</div>', unsafe_allow_html=True)

        frac_zero = lz_bits / 256
        fig_bar = go.Figure(go.Bar(
            x=[frac_zero * 100, (1 - frac_zero) * 100],
            y=[""],
            orientation="h",
            marker_color=["#ef4444", "#22c55e"],
            text=[f"  {lz_bits} zero bits — invalid zone",
                  f"  Valid zone ({(1-frac_zero)*100:.4f}%)"],
            textposition="inside",
            textfont=dict(color="white", size=11, family="IBM Plex Mono"),
            hovertemplate="%{x:.3f}%<extra></extra>",
        ))
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=90,
                              xaxis_title="% of 256-bit space")
        fig_bar.update_layout(margin=dict(t=8, b=28, l=4, r=4))
        fig_bar.update_xaxes(range=[0, 100], showgrid=False)
        fig_bar.update_yaxes(showgrid=False, showticklabels=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown(f"""
        <div style='font-family:"IBM Plex Mono",monospace; font-size:0.70rem;
                    color:#64748b; line-height:2; margin-top:-0.5rem;'>
            <span style='color:#94a3b8;'>bits</span>   {hex(bits)}<br>
            <span style='color:#94a3b8;'>exp </span>   {bits >> 24}<br>
            <span style='color:#94a3b8;'>coef</span>   {hex(bits & 0x00FFFFFF)}<br>
            <span style='color:#94a3b8;'>target</span> {target_hex[:20]}…
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Latest Block Hash</div>', unsafe_allow_html=True)
        bh = block.get("id") or block.get("hash", "N/A")
        zeros = len(bh) - len(bh.lstrip("0"))
        st.code(bh, language=None)
        st.markdown(f'<div style="font-size:.70rem;color:#64748b;">{zeros} leading hex zeros = {zeros*4} leading zero bits</div>',
                    unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel-title">Inter-Block Time Distribution (last 30 blocks)</div>',
                    unsafe_allow_html=True)
        if len(recent) >= 2:
            ts_sorted = sorted([b["timestamp"] for b in recent], reverse=True)
            inter = [ts_sorted[i] - ts_sorted[i+1] for i in range(len(ts_sorted)-1)]

            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(
                x=inter, nbinsx=18, name="Observed",
                marker_color="#3b82f6", opacity=0.8,
                hovertemplate="%{x:.0f} s<extra></extra>",
            ))
            x_th = np.linspace(0, max(inter)*1.2, 300)
            bw   = (max(inter) - min(inter)) / 18
            pdf  = len(inter) * bw * (1/600) * np.exp(-x_th/600)
            fig_h.add_trace(go.Scatter(
                x=x_th, y=pdf, mode="lines", name="Exp(λ=1/600)",
                line=dict(color="#f7931a", width=2, dash="dash"),
            ))
            fig_h.add_vline(x=600, line_dash="dot", line_color="#22c55e",
                            annotation_text="600 s target",
                            annotation_font=dict(color="#22c55e", size=10))
            fig_h.update_layout(
                **PLOTLY_LAYOUT, height=290,
                xaxis_title="seconds between blocks",
                yaxis_title="count",
                legend=dict(orientation="h", y=1.08, font=dict(size=10)),
                bargap=0.05,
            )
            st.plotly_chart(fig_h, use_container_width=True)

            m, med = np.mean(inter), np.median(inter)
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Mean", f"{m:.0f} s")
            sc2.metric("Median", f"{med:.0f} s")
            sc3.metric("Std dev", f"{np.std(inter):.0f} s")

            st.markdown("""
            <div style='font-size:.70rem;color:#64748b;margin-top:.4rem;'>
            Mining is a Bernoulli process → inter-arrival times follow
            <strong style='color:#94a3b8;'>Exp(λ = 1/600 s)</strong>.
            </div>""", unsafe_allow_html=True)
