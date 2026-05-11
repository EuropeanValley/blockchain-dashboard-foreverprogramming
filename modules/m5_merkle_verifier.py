"""
M5 – Merkle Proof Verifier
===========================
Pick any transaction in a block and verify its Merkle inclusion proof
step-by-step, recomputing every hash using hashlib.

Theory
------
Bitcoin organises transactions in a binary hash tree (Merkle tree):
  • Leaves   = SHA256d(txid_bytes)  — but since txids ARE already SHA256d
               hashes of raw transactions, the leaves are simply the txid bytes.
  • Internal  = SHA256d(left_child || right_child)  (32 + 32 = 64 bytes input)
  • Odd level → duplicate the last node before hashing.
  • Root (stored in block header) must equal the recomputed root.

Byte order
----------
Block explorers display txids in *reversed* byte order (big-endian display).
Internally Bitcoin stores them in little-endian order.
Blockstream's API returns txids in display (big-endian) order.
To match the merkle_root in the block header (also returned in display order),
we can work entirely in display/big-endian bytes without reversing — as long
as we are consistent.
"""

import hashlib
import streamlit as st

from api.blockchain_client import (
    get_block, get_block_txids, get_tip_hash,
    mock_latest_block, mock_block_txids,
)

PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#94a3b8", size=11),
    margin=dict(t=10, b=10, l=10, r=10),
)


# ── Merkle helpers ──────────────────────────────────────────────────────────

def sha256d(data: bytes) -> bytes:
    """Double SHA-256: SHA256(SHA256(data))."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def build_merkle_tree(txids: list[str]) -> list[list[str]]:
    """
    Build the full Merkle tree level by level.

    Parameters
    ----------
    txids : list of hex strings (display order, as returned by Blockstream)

    Returns
    -------
    levels : list of levels, each level is a list of hex strings.
             levels[0] = leaves, levels[-1] = [root]
    """
    level = [bytes.fromhex(tx) for tx in txids]
    levels = [[t.hex() for t in level]]

    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])          # duplicate last (Bitcoin rule)
        next_level = []
        for i in range(0, len(level), 2):
            parent = sha256d(level[i] + level[i + 1])
            next_level.append(parent)
        level = next_level
        levels.append([t.hex() for t in level])

    return levels


def merkle_proof(txids: list[str], tx_index: int) -> list[dict]:
    """
    Compute the Merkle inclusion proof for the transaction at tx_index.

    Returns a list of steps:
      { level, index, direction ('left'|'right'), sibling_hash, parent_hash }
    """
    level = [bytes.fromhex(tx) for tx in txids]
    idx   = tx_index
    steps = []

    level_num = 0
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])

        # Which sibling do we pair with?
        if idx % 2 == 0:
            sib_idx  = idx + 1
            direction = "right"
            left, right = level[idx], level[sib_idx]
        else:
            sib_idx  = idx - 1
            direction = "left"
            left, right = level[sib_idx], level[idx]

        parent = sha256d(left + right)
        steps.append({
            "level":       level_num,
            "current":     level[idx].hex(),
            "sibling":     level[sib_idx].hex(),
            "direction":   direction,   # sibling is to the left or right of current
            "left_input":  left.hex(),
            "right_input": right.hex(),
            "parent":      parent.hex(),
        })

        # Move up
        next_level = [sha256d(level[i] + level[i+1]) for i in range(0, len(level), 2)]
        level = next_level
        idx   = idx // 2
        level_num += 1

    return steps


def verify_proof_from_steps(steps: list[dict], claimed_root: str) -> bool:
    """Re-derive the root from the proof path and compare to claimed_root."""
    if not steps:
        return False
    current = bytes.fromhex(steps[0]["current"])
    for s in steps:
        left  = bytes.fromhex(s["left_input"])
        right = bytes.fromhex(s["right_input"])
        current = sha256d(left + right)
    return current.hex() == claimed_root


# ── Render ──────────────────────────────────────────────────────────────────

def render():
    st.markdown("""
    <div style='font-size:.78rem; color:#64748b; margin-bottom:1rem;'>
    Select a block and transaction index to verify its Merkle inclusion proof
    step by step — every SHA-256d computation is shown explicitly.
    </div>""", unsafe_allow_html=True)

    # ── Controls ─────────────────────────────────────────────────────────
    c_hash, c_btn = st.columns([5, 1])
    with c_hash:
        bh_input = st.text_input("Block hash (leave empty for latest)",
                                  placeholder="000000000000000000…",
                                  label_visibility="collapsed", key="m5_hash")
    with c_btn:
        go = st.button("Load →", key="m5_go")

    if not go and "m5_txids" not in st.session_state:
        st.markdown("""<div style='text-align:center; color:#475569;
            font-size:.85rem; margin-top:2rem;'>
            Enter a block hash or click <strong style='color:#f7931a;'>Load →</strong>
            to use the latest block.</div>""", unsafe_allow_html=True)
        return

    if go or "m5_txids" not in st.session_state:
        with st.spinner("Fetching transaction list…"):
            try:
                bh = bh_input.strip() or get_tip_hash()
                block  = get_block(bh)
                txids  = get_block_txids(bh)
                is_mock = False
            except Exception as exc:
                st.warning(f"API unavailable — demo data. ({exc})")
                block   = mock_latest_block()
                txids   = mock_block_txids(n=32)
                is_mock = True

        st.session_state.update(m5_block=block, m5_txids=txids, m5_mock=is_mock)

    block   = st.session_state["m5_block"]
    txids   = st.session_state["m5_txids"]
    is_mock = st.session_state["m5_mock"]

    if is_mock:
        st.caption("⚠️ DEMO DATA — synthetic txids, Merkle root will verify against recomputed root.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Block overview KPIs ───────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="kpi-card"><div class="kpi-label">Block Height</div>
        <div class="kpi-value dim">{block.get('height',0):,}</div></div>""",
        unsafe_allow_html=True)
    c2.markdown(f"""<div class="kpi-card"><div class="kpi-label">Transactions</div>
        <div class="kpi-value dim">{len(txids):,}</div></div>""",
        unsafe_allow_html=True)
    tree_levels = len(txids).bit_length() if len(txids) > 1 else 1
    c3.markdown(f"""<div class="kpi-card"><div class="kpi-label">Tree Depth</div>
        <div class="kpi-value dim">{tree_levels} levels</div></div>""",
        unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Build full tree ───────────────────────────────────────────────────
    if len(txids) > 512:
        st.info("Block has > 512 transactions. Using first 512 for visualization performance.")
        txids = txids[:512]

    levels = build_merkle_tree(txids)
    computed_root = levels[-1][0]
    header_root   = block.get("merkle_root", "")

    # Root verification
    if header_root:
        root_ok = computed_root.lower() == header_root.lower()
        v_color = "#22c55e" if root_ok else "#ef4444"
        v_text  = "✓ ROOT VERIFIED" if root_ok else "✗ ROOT MISMATCH"
    else:
        root_ok = True   # mock path
        v_color, v_text = "#f7931a", "⚠ ROOT NOT CHECKED (demo)"

    st.markdown(f"""
    <div style='font-family:"IBM Plex Mono",monospace; font-size:1.2rem;
                font-weight:700; color:{v_color}; margin-bottom:.75rem;'>
        {v_text}
    </div>""", unsafe_allow_html=True)

    lc, rc = st.columns(2)
    with lc:
        st.markdown('<div style="font-size:.70rem; color:#64748b;">Computed Merkle root:</div>',
                    unsafe_allow_html=True)
        st.code(computed_root, language=None)
    with rc:
        st.markdown('<div style="font-size:.70rem; color:#64748b;">Block header Merkle root:</div>',
                    unsafe_allow_html=True)
        st.code(header_root or "(not available for demo)", language=None)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Transaction picker ────────────────────────────────────────────────
    st.markdown('<div class="panel-title">Step-by-Step Inclusion Proof</div>',
                unsafe_allow_html=True)

    max_idx = len(txids) - 1
    tx_idx  = st.slider(f"Transaction index (0 – {max_idx})", 0, max_idx, 0, key="m5_tx_idx")

    st.markdown(f"""
    <div style='font-family:"IBM Plex Mono",monospace; font-size:.75rem;
                color:#64748b; margin-bottom:.75rem;'>
        Proving inclusion of tx [{tx_idx}]:
        <span style='color:#94a3b8;'>{txids[tx_idx]}</span>
    </div>""", unsafe_allow_html=True)

    proof_steps = merkle_proof(txids, tx_idx)
    verified    = verify_proof_from_steps(proof_steps, computed_root)
    v2_color    = "#22c55e" if verified else "#ef4444"
    v2_text     = "✓ PROOF VALID" if verified else "✗ PROOF INVALID"

    st.markdown(f"""<div style='font-family:"IBM Plex Mono",monospace; font-size:.95rem;
        font-weight:700; color:{v2_color}; margin-bottom:1rem;'>{v2_text}</div>""",
        unsafe_allow_html=True)

    # ── Step-by-step display ─────────────────────────────────────────────
    for i, s in enumerate(proof_steps):
        arrow  = "← sibling left" if s["direction"] == "left" else "sibling right →"
        with st.expander(f"Level {s['level']} → {s['level']+1}  |  {arrow}  |  node index {tx_idx >> i}", expanded=(i==0)):
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"""
                <div style='font-family:"IBM Plex Mono",monospace; font-size:.70rem;
                            line-height:2; color:#64748b;'>
                    <div style='color:#3b82f6; font-weight:700;'>Current hash (level {s['level']}):</div>
                    <div style='color:#94a3b8; word-break:break-all;'>{s['current']}</div>
                    <div style='color:#a855f7; font-weight:700; margin-top:.5rem;'>Sibling hash ({s['direction']}):</div>
                    <div style='color:#94a3b8; word-break:break-all;'>{s['sibling']}</div>
                </div>""", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"""
                <div style='font-family:"IBM Plex Mono",monospace; font-size:.70rem;
                            line-height:2; color:#64748b;'>
                    <div>SHA256d( left || right )</div>
                    <div style='color:#64748b;'>left  = {s['left_input'][:32]}…</div>
                    <div style='color:#64748b;'>right = {s['right_input'][:32]}…</div>
                    <div style='color:#22c55e; font-weight:700; margin-top:.5rem;'>Result (level {s['level']+1}):</div>
                    <div style='color:#22c55e; word-break:break-all;'>{s['parent']}</div>
                </div>""", unsafe_allow_html=True)
            st.code(
                f"import hashlib\n\n"
                f"left  = bytes.fromhex('{s['left_input'][:16]}…')\n"
                f"right = bytes.fromhex('{s['right_input'][:16]}…')\n"
                f"h1 = hashlib.sha256(left + right).digest()\n"
                f"h2 = hashlib.sha256(h1).digest()   # double SHA-256\n"
                f"# = '{s['parent'][:32]}…'",
                language="python",
            )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Tree structure overview ───────────────────────────────────────────
    st.markdown('<div class="panel-title">Merkle Tree Level Sizes</div>',
                unsafe_allow_html=True)
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=[f"Level {i}" for i in range(len(levels))],
        y=[len(l) for l in levels],
        marker_color=["#3b82f6" if i < len(levels)-1 else "#f7931a"
                      for i in range(len(levels))],
        hovertemplate="Level %{x}<br>%{y} nodes<extra></extra>",
    ))
    fig.update_layout(**PL, height=200, yaxis_title="nodes")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Odd-node rule:** when a level has an odd number of nodes, Bitcoin duplicates "
        "the last node before hashing pairs. This is why the node count may not halve "
        "exactly at every level."
    )
