"""
Blockchain API client.

Connects to Blockstream (blockstream.info/api) and Mempool.space (mempool.space/api).
Both APIs are free and require no authentication.

If a request fails (network error, API down, sandbox restriction), the module
falls back to realistic mock data so the dashboard can still run for demo purposes.
"""

import random
import time
from datetime import datetime, timezone

import requests

BLOCKSTREAM_URL = "https://blockstream.info/api"
MEMPOOL_URL = "https://mempool.space/api"
BLOCKCHAIN_INFO_URL = "https://blockchain.info"


def _get(url: str, params: dict | None = None, timeout: int = 10):
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


# ── Live API functions ──────────────────────────────────────────────────────

def get_tip_hash() -> str:
    return _get(f"{BLOCKSTREAM_URL}/blocks/tip/hash").text.strip()


def get_tip_height() -> int:
    return int(_get(f"{BLOCKSTREAM_URL}/blocks/tip/height").text.strip())


def get_block(block_hash: str) -> dict:
    return _get(f"{BLOCKSTREAM_URL}/block/{block_hash}").json()


def get_latest_block() -> dict:
    return get_block(get_tip_hash())


def get_block_header_hex(block_hash: str) -> str:
    """Return the raw 80-byte block header as a lower-case hex string."""
    return _get(f"{BLOCKSTREAM_URL}/block/{block_hash}/header").text.strip()


def get_recent_blocks(n: int = 25) -> list[dict]:
    """Return the n most recent blocks."""
    tip_height = get_tip_height()
    blocks: list[dict] = []
    start_height = tip_height
    while len(blocks) < n:
        batch = _get(f"{BLOCKSTREAM_URL}/blocks/{start_height}").json()
        if not batch:
            break
        blocks.extend(batch)
        start_height = batch[-1]["height"] - 1
    return blocks[:n]


def get_difficulty_history(n_points: int = 100) -> list[dict]:
    """Return difficulty data from blockchain.info. Each entry: {x: timestamp, y: difficulty}."""
    resp = _get(
        f"{BLOCKCHAIN_INFO_URL}/charts/difficulty",
        params={"timespan": "1year", "format": "json", "sampled": "true"},
    )
    return resp.json().get("values", [])[-n_points:]


def get_mempool_fees() -> dict:
    """Current recommended fees in sat/vByte: fastestFee, halfHourFee, hourFee, economyFee."""
    return _get(f"{MEMPOOL_URL}/v1/fees/recommended").json()


def get_mempool_fee_blocks() -> list[dict]:
    return _get(f"{MEMPOOL_URL}/v1/fees/mempool-blocks").json()


def get_blocks_with_fees(start_height: int, count: int = 60) -> list[dict]:
    """Return blocks with fee statistics from Mempool.space for the fee estimator."""
    blocks: list[dict] = []
    height = start_height
    while len(blocks) < count:
        resp = _get(f"{MEMPOOL_URL}/v1/blocks/{height}")
        batch = resp.json()
        if not batch:
            break
        blocks.extend(batch)
        height = batch[-1]["height"] - 1
    return blocks[:count]


# ── Mock / fallback data ────────────────────────────────────────────────────

MOCK_HASH = "0000000000000000000342e9172dc2f26d4a63f2cd38e5a3ae59df28b80756d7"
MOCK_PREV = "00000000000000000002bd1f91c9dcdf9fba84a4c3c3f521fe77e35dcfe51f6b"
MOCK_MERKLE = "a1e2f3d4c5b6a798091827364556473829101112131415161718192021222324"
MOCK_BITS = 0x1703A30C
MOCK_HEIGHT = 895_000


def mock_latest_block() -> dict:
    now = int(time.time())
    return {
        "id": MOCK_HASH,
        "hash": MOCK_HASH,
        "height": MOCK_HEIGHT,
        "version": 0x20000004,
        "timestamp": now - random.randint(30, 600),
        "tx_count": random.randint(1500, 4000),
        "size": random.randint(1_000_000, 1_500_000),
        "weight": random.randint(3_900_000, 4_000_000),
        "merkle_root": MOCK_MERKLE,
        "previousblockhash": MOCK_PREV,
        "bits": MOCK_BITS,
        "nonce": random.randint(0, 0xFFFF_FFFF),
        "difficulty": 113_756_440_312_890.0,
        "mediantime": now - 600,
    }


def mock_recent_blocks(n: int = 25) -> list[dict]:
    """Synthetic blocks with exponentially distributed inter-arrival times (mean=600s)."""
    now = int(time.time())
    blocks, t = [], now
    for i in range(n):
        inter = int(random.expovariate(1 / 600))
        t -= inter
        blocks.append({
            "id": f"mock_{MOCK_HEIGHT - i:07d}",
            "height": MOCK_HEIGHT - i,
            "timestamp": t,
            "tx_count": random.randint(1500, 3500),
            "size": random.randint(900_000, 1_500_000),
            "bits": MOCK_BITS,
            "nonce": random.randint(0, 0xFFFF_FFFF),
            "difficulty": 113_756_440_312_890.0,
        })
    return blocks


def mock_difficulty_history(n_points: int = 100) -> list[dict]:
    base = 1_000_000_000_000.0
    now = int(time.time())
    step = 14 * 24 * 3600 // 10
    points = []
    for i in range(n_points):
        ts = now - (n_points - i) * step
        level = base * (1 + 0.30 * i / n_points + 0.05 * random.gauss(0, 1))
        points.append({"x": ts, "y": max(level, base)})
    return points


def mock_mempool_fees() -> dict:
    base = random.randint(15, 80)
    return {
        "fastestFee": base + random.randint(20, 40),
        "halfHourFee": base + random.randint(5, 20),
        "hourFee": base,
        "economyFee": max(5, base - 10),
        "minimumFee": 1,
    }


def mock_blocks_with_fees(count: int = 60) -> list[dict]:
    now = int(time.time())
    blocks = []
    for i in range(count):
        ts = now - i * 600 - random.randint(-120, 120)
        hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour
        base_fee = 20 + 30 * (1 if 8 <= hour <= 20 else 0) + random.gauss(0, 8)
        tx_count = random.randint(1500, 3500)
        size = tx_count * random.randint(300, 600)
        blocks.append({
            "height": MOCK_HEIGHT - i,
            "timestamp": ts,
            "tx_count": tx_count,
            "size": size,
            "extras": {
                "medianFee": max(1.0, round(base_fee, 2)),
                "feeRange": sorted([
                    max(1, base_fee - 10), max(1, base_fee - 5), base_fee,
                    base_fee + 10, base_fee + 20, base_fee + 40, base_fee + 80,
                ]),
                "totalFees": int(size * base_fee * 0.5),
                "avgFeeRate": round(base_fee, 2),
            },
        })
    return blocks
