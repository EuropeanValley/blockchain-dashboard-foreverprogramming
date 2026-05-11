"""M2 – Block Header Analyzer"""

import hashlib, struct
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from api.blockchain_client import (
    get_block, get_block_header_hex, get_tip_hash, mock_latest_block,
)

PLOTLY_BG = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

def sha256d(data): return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def parse_header(hex_str):
    raw = bytes.fromhex(hex_str)
    assert len(raw) == 80
    return {
        "version":   struct.unpack_from("<I", raw, 0)[0],
        "prev_hash": raw[4:36][::-1].hex(),
        "merkle":    raw[36:68][::-1].hex(),
        "timestamp": struct.unpack_from("<I", raw, 68)[0],
        "bits":      struct.unpack_from("<I", raw, 72)[0],
        "nonce":     struct.unpack_from("<I", raw, 76)[0],
        "raw":       hex_str,
    }

def bits_to_target(bits):
    return (bits & 0x00FF_FFFF) * (2 ** (8 * ((bits >> 24) - 3)))

def verify_pow(hex_str):
    raw      = bytes.fromhex(hex_str)
    h_bytes  = sha256d(raw)
    h_int    = int.from_bytes(h_bytes, "little")
    bits     = struct.unpack_from("<I", raw, 72)[0]
    target   = bits_to_target(bits)
    lz       = 256 - h_int.bit_length() if h_int > 0 else 256
    return dict(hash_display=h_bytes[::-1].hex(), hash_int=h_int,
                target=target, passes=h_int <= target, lz_bits=lz)

def mock_header_hex(block):
    v   = block.get("version", 0x20000004)
    ph  = bytes.fromhex(block.get("previousblockhash", "00"*32))[::-1]
    mr  = bytes.fromhex(block.get("merkle_root", "00"*32))[::-1]
    ts  = block.get("timestamp", 0)
    bi  = block.get("bits", 0x1703A30C)
    no  = block.get("nonce", 0)
    return (struct.pack("<I",v) + ph + mr + struct.pack("<I",ts) +
            struct.pack("<I",bi) + struct.pack("<I",no)).hex()


def render():
    # ── Block selector bar ───────────────────────────────────────────────
    c_inp, c_btn = st.columns([5, 1])
    with c_inp:
        bh_input = st.text_input("Block hash (leave empty for latest)",
                                  placeholder="000000000000000000…", label_visibility="collapsed",
                                  key="m2_hash_input")
    with c_btn:
        go = st.button("Analyze →", key="m2_go")

    if not go and "m2_parsed" not in st.session_state:
        st.markdown("""
        <div style='margin-top:2rem; text-align:center; color:#475569; font-size:.85rem;'>
            Enter a block hash above or click <strong style='color:#f7931a;'>Analyze →</strong>
            to load the latest block.
        </div>""", unsafe_allow_html=True)
        return

    if go or "m2_parsed" not in st.session_state:
        with st.spinner(""):
            try:
                bh = bh_input.strip() or get_tip_hash()
                block = get_block(bh)
                header_hex = get_block_header_hex(bh)
                is_mock = False
            except Exception as exc:
                st.warning(f"API unavailable — showing demo data. ({exc})")
                block = mock_latest_block()
                header_hex = mock_header_hex(block)
                is_mock = True
        st.session_state.update(m2_parsed=parse_header(header_hex),
                                m2_pow=verify_pow(header_hex),
                                m2_mock=is_mock)

    p   = st.session_state["m2_parsed"]
    pow = st.session_state["m2_pow"]
    mock = st.session_state["m2_mock"]

    if mock:
        st.caption("⚠️ DEMO DATA — PoW will not verify (random nonce).")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 1: 6 header fields ───────────────────────────────────────────
    st.markdown('<div class="panel-title">80-Byte Block Header Fields</div>',
                unsafe_allow_html=True)

    dt = datetime.fromtimestamp(p["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    bits = p["bits"]
    exp_v, coef_v = bits >> 24, bits & 0x00FFFFFF
    fields = [
        ("Version",          "4 B", hex(p["version"]),    "Signals which BIP rules apply"),
        ("Timestamp",        "4 B", dt,                    f"Unix epoch: {p['timestamp']}"),
        ("Bits",             "4 B", hex(bits),             f"exp={exp_v}, coef={hex(coef_v)}"),
        ("Nonce",            "4 B", f"{p['nonce']:,}",     "Miner-controlled counter"),
    ]
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, size, value, desc) in zip([c1,c2,c3,c4], fields):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label} <span style='color:#334155;'>({size})</span></div>
            <div class="kpi-value dim" style='font-size:1rem;'>{value}</div>
            <div class="kpi-sub">{desc}</div>
        </div>""", unsafe_allow_html=True)

    # Long hash fields full-width
    for label, size, value, desc in [
        ("Previous Block Hash", "32 B", p["prev_hash"], "SHA-256d of the previous header — the chain link"),
        ("Merkle Root",         "32 B", p["merkle"],    "Root of the Merkle tree of all transactions"),
    ]:
        st.markdown(f"""
        <div class="kpi-card" style='margin-top:.25rem;'>
            <div class="kpi-label">{label} <span style='color:#334155;'>({size}) — little-endian stored, reversed for display</span></div>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:.78rem;
                        color:#94a3b8; word-break:break-all; margin-top:.3rem;'>{value}</div>
            <div class="kpi-sub">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 2: PoW verification ──────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="panel-title">Proof-of-Work Verification</div>',
                    unsafe_allow_html=True)
        verdict_color = "#22c55e" if pow["passes"] else "#ef4444"
        verdict_text  = "✓ VALID" if pow["passes"] else "✗ INVALID (demo)"
        st.markdown(f"""
        <div style='font-family:"IBM Plex Mono",monospace; font-size:1.3rem;
                    font-weight:700; color:{verdict_color}; margin-bottom:1rem;'>
            {verdict_text}
        </div>""", unsafe_allow_html=True)

        lz_hex = len(pow["hash_display"]) - len(pow["hash_display"].lstrip("0"))
        ca, cb = st.columns(2)
        ca.metric("Leading zero bits", f"{pow['lz_bits']} / 256")
        cb.metric("Leading hex zeros", lz_hex)

        st.markdown('<div style="margin-top:.75rem; font-size:.72rem; color:#64748b;">Computed hash (SHA256d):</div>',
                    unsafe_allow_html=True)
        st.code(pow["hash_display"], language=None)
        st.markdown('<div style="font-size:.72rem; color:#64748b;">Target (decoded from bits):</div>',
                    unsafe_allow_html=True)
        st.code(f"{pow['target']:064x}", language=None)

        t_lz = 64 - len(f"{pow['target']:064x}".lstrip("0")) + (64 - len(f"{pow['target']:064x}"))
        st.markdown(f"""
        <div style='font-size:.72rem; color:#64748b; margin-top:.5rem;'>
            hash ≤ target?
            <strong style='color:{verdict_color};'>{"YES ✓" if pow["passes"] else "NO ✗"}</strong>
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel-title">Python Implementation</div>',
                    unsafe_allow_html=True)
        st.code("""\
import hashlib, struct

# 80-byte header in little-endian
raw = bytes.fromhex(header_hex)

# Double SHA-256 (Bitcoin standard)
h1 = hashlib.sha256(raw).digest()
h2 = hashlib.sha256(h1).digest()

# Display order: reverse bytes
hash_display = h2[::-1].hex()

# Numeric comparison: little-endian int
hash_int = int.from_bytes(h2, 'little')

# Decode bits → target
exp  = bits >> 24
coef = bits & 0x00FFFFFF
target = coef * (2 ** (8 * (exp - 3)))

# PoW check
valid = hash_int <= target""", language="python")

        st.markdown("""
        <div style='font-size:.70rem;color:#64748b;margin-top:.6rem;line-height:1.7;'>
        <strong style='color:#94a3b8;'>Byte order note:</strong> All header fields are
        stored in <em>little-endian</em>. Hashes are displayed in reversed (big-endian)
        order on block explorers. The PoW comparison uses the raw little-endian integer.
        </div>""", unsafe_allow_html=True)

    with st.expander("🗂️  Raw 80-byte header (hex)"):
        raw = p["raw"]
        st.markdown(f"""
        <div style='font-family:"IBM Plex Mono",monospace; font-size:.72rem;
                    line-height:2.0; color:#64748b;'>
        <span style='color:#f7931a;'>version  </span> {raw[0:8]}<br>
        <span style='color:#3b82f6;'>prev     </span> {raw[8:72]}<br>
        <span style='color:#22c55e;'>merkle   </span> {raw[72:136]}<br>
        <span style='color:#a855f7;'>timestamp</span> {raw[136:144]}<br>
        <span style='color:#eab308;'>bits     </span> {raw[144:152]}<br>
        <span style='color:#ec4899;'>nonce    </span> {raw[152:160]}
        </div>""", unsafe_allow_html=True)
