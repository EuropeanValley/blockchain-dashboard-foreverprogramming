"""
M6 – Security Score: 51% Attack Cost
======================================
Estimates the real-time USD cost to mount a 51% attack on Bitcoin and
visualises how deeper confirmation depth reduces attack probability.

Theory — Nakamoto (2008) §11
-----------------------------
Suppose an attacker controls fraction q of the network hash rate, while
honest miners control p = 1 − q.  The probability the attacker can
secretly build an alternative chain and overtake the honest chain, given
that the honest chain is already z blocks ahead, is:

    P(z, q) ≈ (q / p)^z          (simplified Gambler's ruin result)

For the exact formula Nakamoto uses a Poisson model for the number of
blocks the attacker can mine while the honest chain extends by z:

    λ = z × (q / p)
    P(z, q) = 1 − Σ_{k=0}^{z}  [Poisson(k; λ) × (1 − (q/p)^{z−k+1})]

Hardware cost model
--------------------
We estimate attack cost using NiceHash SHA-256 rental prices ($/TH/hour).
The attacker needs >50% of current hash rate H (EH/s):

    cost_per_hour = H × 0.51 × 1e6 [TH/s] × price_per_TH_per_hour [$]

Reference: Nakamoto (2008), Bitcoin: A Peer-to-Peer Electronic Cash System
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api.blockchain_client import (
    get_latest_block, get_btc_price_usd,
    mock_latest_block, mock_btc_price,
)

PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#94a3b8", size=11),
    margin=dict(t=36, b=36, l=8, r=8),
    xaxis=dict(gridcolor="#1e2535", linecolor="#1e2535", zeroline=False),
    yaxis=dict(gridcolor="#1e2535", linecolor="#1e2535", zeroline=False),
)

# NiceHash SHA-256 rental price (conservative estimate, USD / TH / hour)
NICEHASH_PRICE_USD_PER_TH_HOUR = 0.0654   # approx current rate


# ── Attack probability (Nakamoto 2008 §11) ──────────────────────────────────

def attack_probability_exact(q: float, z: int) -> float:
    """
    Exact Nakamoto formula for P(attacker overtakes from z blocks behind).

    Uses Poisson distribution of attacker's progress during honest z-block lead.
    """
    if q >= 0.5:
        return 1.0
    p = 1.0 - q
    lam = z * (q / p)

    # Σ_{k=0}^{z} Poisson(k; λ) × (1 − (q/p)^(z−k+1))
    total = 0.0
    for k in range(z + 1):
        # log-space Poisson PMF to avoid overflow for large z
        log_pmf = k * math.log(lam) - lam - sum(math.log(j) for j in range(1, k+1))
        pmf = math.exp(log_pmf)
        if k < z:
            total += pmf * (1.0 - (q / p) ** (z - k + 1))
        else:
            total += pmf
    return max(0.0, 1.0 - total)


def attack_probability_simple(q: float, z: int) -> float:
    """Simplified Gambler's ruin: (q/p)^z  for q < p."""
    if q >= 0.5:
        return 1.0
    return (q / (1 - q)) ** z


# ── Cost model ──────────────────────────────────────────────────────────────

def bits_to_difficulty(bits: int) -> float:
    target   = (bits & 0x00FF_FFFF) * (2 ** (8 * ((bits >> 24) - 3)))
    genesis  = 0x00FFFF * (2 ** (8 * (0x1D - 3)))
    return genesis / target


def hashrate_from_difficulty(diff: float) -> float:
    """Estimated network hash rate in H/s."""
    return diff * (2 ** 32) / 600


def attack_cost_usd_per_hour(hashrate_hs: float, attacker_fraction: float = 0.51) -> float:
    """
    Estimated USD / hour to rent attacker_fraction of current hash rate.
    hashrate_hs: total network hash rate in H/s
    """
    attacker_ths = hashrate_hs * attacker_fraction / 1e12   # TH/s
    return attacker_ths * NICEHASH_PRICE_USD_PER_TH_HOUR


def btc_blocks_per_hour(btc_price: float, attack_cost: float) -> float:
    """How many BTC-worth-of-block-rewards could cover the attack per hour."""
    block_reward_btc = 3.125   # post-4th-halving (April 2024)
    block_reward_usd = block_reward_btc * btc_price
    blocks_per_hour  = 6
    revenue_per_hour = block_reward_usd * blocks_per_hour
    return revenue_per_hour / attack_cost if attack_cost > 0 else 0


# ── Render ──────────────────────────────────────────────────────────────────

def render():
    st.markdown("""
    <div style='font-size:.78rem; color:#64748b; margin-bottom:1rem;'>
    Estimates the real-time USD cost to execute a 51% attack and visualises
    how confirmation depth reduces attack success probability (Nakamoto 2008, §11).
    </div>""", unsafe_allow_html=True)

    c_btn, _ = st.columns([1, 4])
    with c_btn:
        refresh_btn = st.button("↺  Refresh", key="m6_go")

    if refresh_btn or "m6_block" not in st.session_state:
        with st.spinner("Fetching network data…"):
            try:
                block     = get_latest_block()
                btc_price = get_btc_price_usd()
                mock      = False
            except Exception as exc:
                st.warning(f"API unavailable — demo data. ({exc})")
                block     = mock_latest_block()
                btc_price = mock_btc_price()
                mock      = True
        st.session_state.update(m6_block=block, m6_price=btc_price, m6_mock=mock)

    block     = st.session_state["m6_block"]
    btc_price = st.session_state["m6_price"]
    mock      = st.session_state["m6_mock"]

    if mock:
        st.caption("⚠️ DEMO DATA — estimated values.")

    # ── Derived values ────────────────────────────────────────────────────
    bits       = block.get("bits", 0x1703A30C)
    difficulty = bits_to_difficulty(bits)
    hashrate   = hashrate_from_difficulty(difficulty)
    hr_ehs     = hashrate / 1e18

    attack_51_cost = attack_cost_usd_per_hour(hashrate, 0.51)
    attack_33_cost = attack_cost_usd_per_hour(hashrate, 0.34)
    revenue_ratio  = btc_blocks_per_hour(btc_price, attack_51_cost)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Network Hash Rate",          f"{hr_ehs:.2f} EH/s",              "blue"),
        ("51% Attack Cost",            f"${attack_51_cost/1e6:.1f}M / hr", ""),
        ("BTC Price",                  f"${btc_price:,.0f}",               "dim"),
        ("Block Revenue / Attack Cost",f"{revenue_ratio:.4f}×",           "dim"),
    ]
    for col, (label, val, cls) in zip([c1,c2,c3,c4], kpis):
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {cls}">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="panel-title">Attack Success Probability vs Confirmation Depth</div>',
                    unsafe_allow_html=True)

        z_vals = list(range(1, 31))
        attacker_fractions = [0.10, 0.20, 0.30, 0.40, 0.49]
        colors = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"]

        fig = go.Figure()
        for q, color in zip(attacker_fractions, colors):
            probs = [attack_probability_exact(q, z) for z in z_vals]
            fig.add_trace(go.Scatter(
                x=z_vals, y=[p * 100 for p in probs],
                mode="lines+markers", name=f"q = {int(q*100)}%",
                line=dict(color=color, width=2),
                marker=dict(size=5),
                hovertemplate=f"q={int(q*100)}% | z=%{{x}}<br>P=%{{y:.3f}}%<extra></extra>",
            ))

        # 1% and 0.1% threshold lines
        fig.add_hline(y=1.0,  line_dash="dot", line_color="#334155",
                      annotation_text="1%",  annotation_font=dict(size=9, color="#64748b"))
        fig.add_hline(y=0.1,  line_dash="dot", line_color="#1e2535",
                      annotation_text="0.1%", annotation_font=dict(size=9, color="#64748b"))

        # "6 confirmations" reference
        fig.add_vline(x=6, line_dash="dash", line_color="#f7931a",
                      annotation_text="6 confs", annotation_font=dict(color="#f7931a", size=10))

        fig.update_layout(**PL, height=360,
                          xaxis_title="confirmation depth (z)",
                          yaxis_title="P(attack succeeds) %",
                          yaxis_type="log",
                          legend=dict(orientation="h", y=1.08, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="panel-title">Attack Cost by Hash Rate Fraction</div>',
                    unsafe_allow_html=True)

        fractions = np.linspace(0.05, 0.70, 66)
        costs_mln = [attack_cost_usd_per_hour(hashrate, f) / 1e6 for f in fractions]
        fig2 = go.Figure(go.Scatter(
            x=fractions * 100, y=costs_mln,
            mode="lines", fill="tozeroy",
            line=dict(color="#ef4444", width=2),
            fillcolor="rgba(239,68,68,0.08)",
            hovertemplate="q = %{x:.1f}%<br>$%{y:.1f}M / hr<extra></extra>",
        ))
        fig2.add_vline(x=51, line_dash="dash", line_color="#f7931a",
                       annotation_text="51%", annotation_font=dict(color="#f7931a", size=10))
        fig2.update_layout(**PL, height=200,
                           xaxis_title="attacker hash rate fraction (%)",
                           yaxis_title="$M / hour")
        st.plotly_chart(fig2, use_container_width=True)

        # Confirmation table
        st.markdown('<div class="panel-title" style="margin-top:.5rem;">P(success) for q = 30%</div>',
                    unsafe_allow_html=True)
        rows = [(z, f"{attack_probability_exact(0.30, z)*100:.4f}%",
                    f"{attack_probability_simple(0.30, z)*100:.4f}%")
                for z in [1, 2, 3, 6, 10, 20, 30]]
        tbl = pd.DataFrame(rows, columns=["z (confs)", "Exact (§11)", "Simple (q/p)^z"])
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Cost model detail ─────────────────────────────────────────────────
    st.markdown('<div class="panel-title">Cost Model Breakdown</div>',
                unsafe_allow_html=True)

    cl, cr = st.columns(2, gap="large")
    with cl:
        st.markdown(f"""
        <div style='font-family:"IBM Plex Mono",monospace; font-size:.75rem;
                    line-height:2.2; color:#64748b;'>
        <span style='color:#94a3b8;'>Network hash rate      </span>  {hr_ehs:.2f} EH/s<br>
        <span style='color:#94a3b8;'>51% threshold          </span>  {hr_ehs*0.51:.2f} EH/s<br>
        <span style='color:#94a3b8;'>NiceHash SHA-256 rate  </span>  ${NICEHASH_PRICE_USD_PER_TH_HOUR:.4f} / TH / hr<br>
        <span style='color:#94a3b8;'>Attacker TH/s needed   </span>  {hashrate*0.51/1e12:,.0f} TH/s<br>
        <span style='color:#f7931a;'>51% cost / hour        </span>  ${attack_51_cost:,.0f}<br>
        <span style='color:#64748b;'>34% cost / hour        </span>  ${attack_33_cost:,.0f}<br>
        <span style='color:#94a3b8;'>Block reward (3.125 BTC)</span> ${3.125 * btc_price:,.0f}<br>
        <span style='color:#94a3b8;'>Revenue 6 blocks/hr    </span> ${3.125 * btc_price * 6:,.0f}
        </div>""", unsafe_allow_html=True)

    with cr:
        st.markdown(r"""
**Nakamoto (2008) §11 formula:**

$$P(z, q) = 1 - \sum_{k=0}^{z}
    \frac{\lambda^k e^{-\lambda}}{k!}
    \left(1 - \left(\frac{q}{p}\right)^{z-k+1}\right)$$

where $\lambda = z \cdot \frac{q}{p}$,  $p = 1 - q$.

The log-scale chart above shows that with **6 confirmations**,
even an attacker with 30% of hash rate has < 1% success probability.
At 49%, 6 confirmations still give ~50% — requiring many more.
        """)

    st.info(
        "**Cost model assumptions:** NiceHash SHA-256 rental price "
        f"${NICEHASH_PRICE_USD_PER_TH_HOUR:.4f}/TH/hr (market rate, varies). "
        "Actual attack would also require acquiring hardware over time, "
        "making a sustained attack even more expensive in practice."
    )
