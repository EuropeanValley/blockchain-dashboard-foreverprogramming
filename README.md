[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/N3kLi3ZO)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23659097&assignment_repo_type=AssignmentRepo)

# CryptoChain Analyzer Dashboard

Real-time Bitcoin cryptographic metrics dashboard with AI-powered fee estimation.

## Student Information

| Field | Value |
|---|---|
| Student Name | Zihao Ying |
| GitHub Username | foreverprogramming |
| Project Title | CryptoChain Analyzer Dashboard |
| Chosen AI Approach | Fee Estimator (Gradient Boosting Regressor, sat/vByte) |

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard auto-fetches live data. If the APIs are unreachable it falls back to
realistic demo data so all modules remain usable.

## Module Tracking

| Module | What it includes | Status |
|---|---|---|
| M1 | Proof of Work Monitor | Done |
| M2 | Block Header Analyzer | Done |
| M3 | Difficulty History | Done |
| M4 | AI Fee Estimator | In progress |

## Current Progress (Checkpoint – 29 April 2026)

- **M1**: Live difficulty value decoded from `bits` field. Leading-zero threshold
  visualisation in 256-bit space. Inter-block time histogram with exponential
  distribution overlay. Estimated network hash rate in EH/s. Auto-refreshes every 60 s.
- **M2**: Full 80-byte header parsed (all 6 fields, little-endian aware).
  Local `SHA256(SHA256(header))` verification with `hashlib`. Leading-zero bit count.
  Target decoded from `bits` and compared against computed hash.
- **M3**: Difficulty history chart (log scale) from blockchain.info API.
  Adjustment events detected and marked. Block-time ratio per epoch plotted.
  Adjustment formula explained with LaTeX.
- **M4**: Gradient Boosting Regressor skeleton complete. Feature engineering
  (`hour`, `day_of_week`, `tx_count`, `size_mb`, `fullness`, `lag_fee`).
  Train/test split, MAE & R² evaluation, feature importance chart,
  interactive live prediction widget, comparison with Mempool.space recommendations.

## Next Step

- Polish M4: integrate real-time mempool depth as an additional feature via
  Mempool.space WebSocket stream.
- Add optional M5 (Merkle Proof Verifier).
- Write final PDF report.

## Main Problem or Blocker

- Mempool.space `/v1/blocks` endpoint returns fee extras only for recent blocks;
  older blocks may lack `medianFee`. Workaround: fall back to `avgFeeRate`.

## APIs Used

| API | URL | Used for |
|---|---|---|
| Blockstream | `blockstream.info/api` | Block data, headers, recent blocks |
| Mempool.space | `mempool.space/api` | Fee recommendations, blocks with fee stats |
| Blockchain.info | `blockchain.info` | Difficulty history chart |

## Project Structure

```text
blockchain-dashboard-foreverprogramming/
├── README.md
├── requirements.txt
├── app.py                          ← Dashboard entry point
├── api/
│   ├── __init__.py
│   └── blockchain_client.py       ← All API calls + mock fallback data
└── modules/
    ├── __init__.py
    ├── m1_pow_monitor.py           ← PoW Monitor (M1)
    ├── m2_block_header.py          ← Block Header Analyzer (M2)
    ├── m3_difficulty_history.py    ← Difficulty History (M3)
    └── m4_ai_component.py          ← Fee Estimator AI (M4)
```

## Cryptographic Concepts Applied

- **SHA-256d**: `SHA256(SHA256(data))` — Bitcoin's standard double hash, used
  in block header PoW verification (M2).
- **Proof of Work**: Valid block hash ≤ target threshold. Verified locally with
  `hashlib` (M2).
- **bits → target decoding**: `target = coefficient × 2^(8×(exponent−3))` (M1, M2).
- **Difficulty formula**: `difficulty = genesis_target / current_target` (M1, M3).
- **Hash rate estimate**: `hashrate ≈ difficulty × 2^32 / 600` (M1).
- **Merkle trees**: Root stored in block header; used as data integrity proof (M2 display).
- **Exponential inter-arrival**: Mining is a Bernoulli process → block times are
  geometrically distributed, approximated by Exp(λ = 1/600) (M1 histogram).
