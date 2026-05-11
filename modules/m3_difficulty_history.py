"""M3 – Difficulty History"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api.blockchain_client import get_difficulty_history, mock_difficulty_history

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#94a3b8", size=11),
    margin=dict(t=36, b=36, l=8, r=8),
    xaxis=dict(gridcolor="#1e2535", linecolor="#1e2535", zeroline=False),
    yaxis=dict(gridcolor="#1e2535", linecolor="#1e2535", zeroline=False),
)


def _load(n):
    try:
        vals = get_difficulty_history(n)
        return vals, False
    except Exception as exc:
        return mock_difficulty_history(n), True


def render():
    # ── Controls ─────────────────────────────────────────────────────────
    c_sl, c_btn = st.columns([4, 1])
    with c_sl:
        n = st.slider("Data points", 20, 365, 120, key="m3_n", label_visibility="collapsed")
    with c_btn:
        load = st.button("Load →", key="m3_load")

    if load or "m3_df" not in st.session_state or st.session_state.get("m3_n_prev") != n:
        with st.spinner(""):
            vals, mock = _load(n)
        df = pd.DataFrame(vals)
        df["Date"] = pd.to_datetime(df["x"], unit="s", utc=True)
        df = df.rename(columns={"y": "Difficulty"}).sort_values("Date")
        df["pct"]   = df["Difficulty"].pct_change()
        df["ratio"] = df["Difficulty"].shift(1) / df["Difficulty"]
        df["adj"]   = df["pct"].abs() > 0.01
        st.session_state.update(m3_df=df, m3_mock=mock, m3_n_prev=n)

    df   = st.session_state["m3_df"]
    mock = st.session_state["m3_mock"]

    if mock:
        st.caption("⚠️ DEMO DATA")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────
    latest  = df["Difficulty"].iloc[-1]
    first   = df["Difficulty"].iloc[0]
    change  = (latest / first - 1) * 100
    n_adj   = int(df["adj"].sum())
    avg_rat = df["ratio"].dropna().mean()

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Current Difficulty",   f"{latest/1e12:.2f} T",   ""),
        (c2, "Period Change",        f"{change:+.1f}%",         "green" if change >= 0 else ""),
        (c3, "Adjustment Events",    str(n_adj),                "dim"),
        (c4, "Avg Block-Time Ratio", f"{avg_rat:.4f}×",         "blue"),
    ]
    for col, label, val, cls in cards:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {cls}">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Charts side by side ───────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="panel-title">Difficulty Over Time (log scale)</div>',
                    unsafe_allow_html=True)
        adj_df = df[df["adj"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Difficulty"],
            mode="lines", name="Difficulty",
            line=dict(color="#3b82f6", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=adj_df["Date"], y=adj_df["Difficulty"],
            mode="markers", name="Adjustment",
            marker=dict(color="#f7931a", size=9, symbol="diamond",
                        line=dict(color="#fff", width=1)),
            hovertemplate="Adjustment<br>%{x|%Y-%m-%d}<extra></extra>",
        ))
        fig.update_yaxes(type="log", title="Difficulty (log₁₀)")
        fig.update_layout(**PLOTLY_LAYOUT, height=340,
                          legend=dict(orientation="h", y=1.08, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="panel-title">Block-Time Ratio per Epoch</div>',
                    unsafe_allow_html=True)
        r_df = df.dropna(subset=["ratio"])
        colors = ["#22c55e" if r < 1 else "#ef4444" for r in r_df["ratio"]]
        fig2 = go.Figure(go.Bar(
            x=r_df["Date"], y=r_df["ratio"],
            marker_color=colors, opacity=0.85,
            hovertemplate="%{x|%Y-%m-%d}<br>ratio %{y:.4f}<extra></extra>",
        ))
        fig2.add_hline(y=1.0,  line_dash="dash", line_color="#64748b",
                       annotation_text="target", annotation_font=dict(size=9))
        fig2.add_hline(y=0.25, line_dash="dot",  line_color="#334155")
        fig2.add_hline(y=4.0,  line_dash="dot",  line_color="#334155")
        fig2.update_layout(**PLOTLY_LAYOUT, height=340,
                           yaxis_title="D_old / D_new")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Formula explainer ─────────────────────────────────────────────────
    st.markdown('<div class="panel-title">Difficulty Adjustment Formula</div>',
                unsafe_allow_html=True)
    cl, cr = st.columns([2, 1])
    with cl:
        st.markdown(r"""
$$\text{new\_difficulty} = \text{old\_difficulty}
\times \frac{\text{actual time (2016 blocks)}}
{2016 \times 600\text{ s}}$$

Ratio clamped to **[¼ , 4]** to prevent extreme swings.
        """)
    with cr:
        st.markdown(f"""
        <div style='font-family:"IBM Plex Mono",monospace; font-size:.75rem;
                    color:#64748b; line-height:2;'>
        <span style='color:#94a3b8;'>epoch blocks</span>  2,016<br>
        <span style='color:#94a3b8;'>target time  </span>  1,209,600 s<br>
        <span style='color:#94a3b8;'>floor ratio  </span>  × 0.25<br>
        <span style='color:#94a3b8;'>ceiling ratio</span>  × 4.00
        </div>""", unsafe_allow_html=True)
