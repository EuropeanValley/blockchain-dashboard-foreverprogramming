"""
M2 – Block Header Analyzer
============================
Displays the 80-byte Bitcoin block header and verifies the Proof of Work
locally using Python's hashlib, without relying on any external library.

Bitcoin block header layout (80 bytes, all fields in LITTLE-ENDIAN):
  ┌─────────────────────────────────────────────────────────────────┐
  │ Offset │ Size │ Field           │ Description                   │
  ├────────┼──────┼─────────────────┼───────────────────────────────┤
  │ 0      │ 4 B  │ Version         │ Block version number          │
  │ 4      │ 32 B │ Prev block hash │ SHA-256d of previous header   │
  │ 36     │ 32 B │ Merkle root     │ Root of the transaction tree  │
  │ 68     │ 4 B  │ Timestamp       │ Unix epoch (seconds)          │
  │ 72     │ 4 B  │ Bits            │ Compact target encoding       │
  │ 76     │ 4 B  │ Nonce           │ Miner-controlled counter      │
  └─────────────────────────────────────────────────────────────────┘
"""

import hashlib
import struct
from datetime import datetime, timezone

import streamlit as st

from api.blockchain_client import (
    get_block,
    get_block_header_hex,
    get_latest_block,
    get_tip_hash,
    mock_latest_block,
)

# ── Cryptographic helpers ───────────────────────────────────────────────────

def sha256d(data: bytes) -> bytes:
    """Double SHA-256: SHA256(SHA256(data)). This is Bitcoin's standard hash."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def parse_header(header_hex: str) -> dict:
    """
    Parse the 80-byte header from its hex representation.

    Bitcoin stores multi-byte integers in little-endian order.
    Hashes are stored in their internal byte order (which is reversed
    compared to the display order used on block explorers).
    """
    raw = bytes.fromhex(header_hex)
    assert len(raw) == 80, f"Expected 80 bytes, got {len(raw)}"

    version    = struct.unpack_from("<I", raw, 0)[0]       # 4 B LE uint32
    prev_hash  = raw[4:36][::-1].hex()                     # 32 B, reverse for display
    merkle     = raw[36:68][::-1].hex()                    # 32 B, reverse for display
    timestamp  = struct.unpack_from("<I", raw, 68)[0]      # 4 B LE uint32
    bits       = struct.unpack_from("<I", raw, 72)[0]      # 4 B LE uint32
    nonce      = struct.unpack_from("<I", raw, 76)[0]      # 4 B LE uint32

    return {
        "version": version,
        "prev_hash": prev_hash,
        "merkle_root": merkle,
        "timestamp": timestamp,
        "bits": bits,
        "nonce": nonce,
        "raw_hex": header_hex,
    }


def bits_to_target(bits: int) -> int:
    """Decode compact 'bits' encoding → full 256-bit target integer."""
    exp = bits >> 24
    coef = bits & 0x00FF_FFFF
    return coef * (2 ** (8 * (exp - 3)))


def verify_pow(header_hex: str) -> dict:
    """
    Compute SHA256(SHA256(header)) and check it is ≤ target.

    Returns a dict with hash (display order), target, passes, leading_zero_bits.
    """
    raw = bytes.fromhex(header_hex)
    # Double-SHA256 of the 80-byte header
    hash_bytes_le = sha256d(raw)          # little-endian (internal order)
    hash_bytes_be = hash_bytes_le[::-1]   # big-endian  (display order, used by explorers)
    hash_int = int.from_bytes(hash_bytes_le, "little")   # interpret as 256-bit integer

    bits = struct.unpack_from("<I", raw, 72)[0]
    target = bits_to_target(bits)

    passes = hash_int <= target
    lz_bits = 256 - hash_int.bit_length() if hash_int > 0 else 256

    return {
        "hash_display": hash_bytes_be.hex(),   # as shown on block explorers
        "hash_int": hash_int,
        "target": target,
        "passes": passes,
        "leading_zero_bits": lz_bits,
    }


def make_mock_header_hex(block: dict) -> str:
    """
    Construct a plausible 80-byte header hex from block metadata.
    The hash will NOT pass PoW (nonce is arbitrary), but the structure is correct.
    Used only when the API /header endpoint is unavailable.
    """
    version = block.get("version", 0x20000004)
    prev_raw = bytes.fromhex(block.get("previousblockhash", "00" * 32))[::-1]
    merkle_raw = bytes.fromhex(block.get("merkle_root", "00" * 32))[::-1]
    timestamp = block.get("timestamp", 0)
    bits = block.get("bits", 0x1703A30C)
    nonce = block.get("nonce", 0)

    header = (
        struct.pack("<I", version) +
        prev_raw +
        merkle_raw +
        struct.pack("<I", timestamp) +
        struct.pack("<I", bits) +
        struct.pack("<I", nonce)
    )
    return header.hex()


# ── Render ──────────────────────────────────────────────────────────────────

def render() -> None:
    st.header("🔍 M2 – Block Header Analyzer")
    st.markdown(
        "Inspect the 80-byte Bitcoin block header and **verify the Proof of Work "
        "locally** using `hashlib.sha256`. No external crypto library used."
    )

    # ── Block selection ───────────────────────────────────────────────────
    col_a, col_b = st.columns([3, 1])
    with col_a:
        block_hash_input = st.text_input(
            "Block hash (leave empty to use the latest block)",
            placeholder="000000000000000000…",
            key="m2_hash_input",
        )
    with col_b:
        lookup = st.button("🔎 Analyze", key="m2_lookup")

    if not lookup:
        st.info("Enter a block hash or click **Analyze** to load the latest block.")
        return

    with st.spinner("Fetching block data…"):
        is_mock = False
        try:
            if block_hash_input.strip():
                bh = block_hash_input.strip()
                block = get_block(bh)
                header_hex = get_block_header_hex(bh)
            else:
                bh = get_tip_hash()
                block = get_block(bh)
                header_hex = get_block_header_hex(bh)
        except Exception as exc:
            st.warning(f"Live API unavailable ({exc}). Showing demo data.")
            block = mock_latest_block()
            header_hex = make_mock_header_hex(block)
            is_mock = True

    if is_mock:
        st.caption("⚠️ DEMO DATA — constructed locally, PoW will NOT verify (nonce is random).")

    # ── Parse ─────────────────────────────────────────────────────────────
    parsed = parse_header(header_hex)
    pow_result = verify_pow(header_hex)
    target = pow_result["target"]
    bits = parsed["bits"]
    exp_val = bits >> 24
    coef_val = bits & 0x00FF_FFFF

    st.divider()
    st.subheader("📦 Block Header Fields (80 bytes)")

    header_data = {
        "Field": [
            "Version", "Previous Block Hash",
            "Merkle Root", "Timestamp", "Bits", "Nonce"
        ],
        "Bytes": ["4", "32", "32", "4", "4", "4"],
        "Value": [
            hex(parsed["version"]),
            parsed["prev_hash"],
            parsed["merkle_root"],
            f"{parsed['timestamp']} → {datetime.fromtimestamp(parsed['timestamp'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            hex(bits),
            f"{parsed['nonce']:,}",
        ],
        "Description": [
            "Signals which BIP rules this block follows",
            "SHA-256d of the previous block header (chain link)",
            "Root of the Merkle tree of all transactions in this block",
            "Unix epoch (seconds since 1970-01-01 00:00:00 UTC)",
            f"Compact target: exp={exp_val}, coef={hex(coef_val)}",
            "32-bit counter varied by miners until hash ≤ target",
        ],
    }
    import pandas as pd
    df_header = pd.DataFrame(header_data)
    st.dataframe(df_header, use_container_width=True, hide_index=True)

    # ── Raw header hex ────────────────────────────────────────────────────
    with st.expander("🗂️ Raw 80-byte header (hex)"):
        raw_hex = parsed["raw_hex"]
        annotated = (
            f"**Version (4 B):** `{raw_hex[0:8]}`  \n"
            f"**Prev hash (32 B):** `{raw_hex[8:72]}`  \n"
            f"**Merkle root (32 B):** `{raw_hex[72:136]}`  \n"
            f"**Timestamp (4 B):** `{raw_hex[136:144]}`  \n"
            f"**Bits (4 B):** `{raw_hex[144:152]}`  \n"
            f"**Nonce (4 B):** `{raw_hex[152:160]}`"
        )
        st.markdown(annotated)
        st.code(raw_hex, language=None)

    st.divider()
    st.subheader("✅ Proof-of-Work Verification")
    st.markdown(
        "**Algorithm:** `SHA256(SHA256(header_bytes))` using Python's `hashlib`. "
        "The result must be ≤ the target encoded in the `bits` field."
    )

    col_code, col_result = st.columns([2, 1])

    with col_code:
        st.code(
            "import hashlib, struct\n\n"
            "raw = bytes.fromhex(header_hex)          # 80 bytes\n"
            "h1  = hashlib.sha256(raw).digest()       # first SHA-256\n"
            "h2  = hashlib.sha256(h1).digest()        # second SHA-256\n"
            "# Display order: reverse byte order\n"
            "hash_display = h2[::-1].hex()\n"
            "# Numeric comparison is in little-endian\n"
            "hash_int = int.from_bytes(h2, 'little')\n"
            "valid = hash_int <= target",
            language="python",
        )

    with col_result:
        hash_display = pow_result["hash_display"]
        passes = pow_result["passes"]
        lz = pow_result["leading_zero_bits"]

        verdict = "🟢 VALID" if passes else "🔴 INVALID (expected for demo)"
        st.markdown(f"### {verdict}")
        st.metric("Leading zero bits", f"{lz} / 256")
        lz_hex = len(hash_display) - len(hash_display.lstrip("0"))
        st.metric("Leading hex zeros", lz_hex)

    st.markdown("**Computed hash (SHA256d):**")
    st.code(hash_display, language=None)

    target_hex = f"{target:064x}"
    st.markdown("**Target (decoded from bits):**")
    st.code(target_hex, language=None)

    # Colour-coded comparison
    lz_hex_t = len(target_hex) - len(target_hex.lstrip("0"))
    st.markdown(
        f"Leading zeros in hash: **{lz_hex}** hex chars | "
        f"Leading zeros in target: **{lz_hex_t}** hex chars  \n"
        f"Hash ≤ Target? → **{'YES ✅' if passes else 'NO ❌'}**"
    )

    st.info(
        "**Byte-order note:** Bitcoin stores all header fields in little-endian. "
        "When displaying the hash or previous-block hash on block explorers, "
        "the bytes are reversed (big-endian display order). "
        "The PoW comparison is done on the raw little-endian integer."
    )
