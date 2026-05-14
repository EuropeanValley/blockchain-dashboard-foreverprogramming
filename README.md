---
noteId: "16616e004ebf11f1b47d6521b6fe6a69"
tags: []

---

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/N3kLi3ZO)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23659097&assignment_repo_type=AssignmentRepo)

# CryptoChain Analyzer Dashboard

Real-time Bitcoin cryptographic metrics dashboard with AI-powered fee estimation and statistical anomaly detection.

## Student Information

| Field | Value |
|---|---|
| Student Name | Zihao Ying |
| GitHub Username | foreverprogramming |
| Project Title | CryptoChain Analyzer Dashboard |
| AI Approach (M4) | Fee Estimator — Gradient Boosting Regressor (sat/vByte) |
| AI Approach (M7) | Block Anomaly Detector — Exponential distribution baseline (Z-score + p-value) |

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. All modules auto-fetch live data from public APIs (no key required). If any API is unreachable, every module falls back gracefully to realistic demo data so the dashboard remains fully usable.

## Module Tracking

| Module | Description | Status |
|---|---|---|
| M1 | Proof of Work Monitor | ✅ Complete |
| M2 | Block Header Analyzer | ✅ Complete |
| M3 | Difficulty History | ✅ Complete |
| M4 | AI Fee Estimator (Gradient Boosting) | ✅ Complete |
| M5 | Merkle Proof Verifier *(optional)* | ✅ Complete |
| M6 | Security Score — 51% Attack Cost *(optional)* | ✅ Complete |
| M7 | Block Anomaly Detector — 2nd AI *(optional)* | ✅ Complete |

## Current Progress

**M1 — Proof of Work Monitor**
- Decodes the `bits` field into the full 256-bit target using `target = coef × 2^(8×(exp−3))`.
- Visualises the SHA-256 space as a horizontal bar showing the fraction of invalid vs valid hashes.
- Inter-block time histogram with a theoretical Exp(λ = 1/600) overlay and explanation of the Poisson process model.
- Network hash rate estimated from `difficulty × 2³² / 600`.
- Auto-refreshes every 60 seconds.

**M2 — Block Header Analyzer**
- Parses all 6 fields of the 80-byte header (version, prev_hash, merkle_root, timestamp, bits, nonce) with correct little-endian byte handling.
- Locally computes `SHA256(SHA256(header))` using only `hashlib` and verifies the result is ≤ target.
- Annotated raw hex view with each field colour-coded.

**M3 — Difficulty History**
- Log-scale difficulty chart from blockchain.info with adjustment events detected and marked (>1% change threshold).
- Bar chart of block-time ratio (D_old / D_new) per sampled epoch with protocol floor (×0.25) and ceiling (×4) reference lines.
- LaTeX adjustment formula with worked example.

**M4 — AI Fee Estimator**
- Gradient Boosting Regressor trained on Mempool.space block-level data.
- Features: `hour`, `day_of_week`, `tx_count`, `size_mb`, `fullness`, `lag_fee`, `lag_fee2`, `roll_mean`.
- Evaluation: MAE, RMSE, MAPE, R², plus 3-fold `TimeSeriesSplit` cross-validation (no shuffle, respects temporal order).
- Charts: actual vs predicted scatter, residual distribution with Normal overlay, feature importance, predicted vs actual over time.
- Live prediction widget with fee priority tier classifier (LOW / MEDIUM / HIGH / PRIORITY) and side-by-side comparison with Mempool.space recommendations.

**M5 — Merkle Proof Verifier** *(optional)*
- Builds the full Merkle tree from a block's transaction list using `SHA256d`.
- User picks any transaction by index; the step-by-step proof path is shown with every hash input and output rendered explicitly.
- Handles Bitcoin's odd-node duplication rule. Recomputed root is verified against the block header's `merkle_root`.

**M6 — Security Score** *(optional)*
- Derives current network hash rate from the `bits` field (no extra API call needed).
- Estimates USD/hour cost to rent 51% of hash rate using NiceHash SHA-256 pricing.
- Implements the exact Nakamoto (2008) §11 Poisson formula for P(attack succeeds | z confirmations, attacker fraction q) alongside the simplified Gambler's ruin approximation, with a side-by-side comparison table.
- Log-scale chart showing P(success) for attacker fractions 10–49% across 1–30 confirmations.

**M7 — Block Anomaly Detector** *(optional, second AI approach)*
- Unsupervised statistical detector using an Exponential(λ = 1/600) baseline.
- Two complementary signals per block: two-sided p-value against the Exp CDF, and Z-score on log-transformed inter-arrival times (Gumbel-distributed).
- Timeline, histogram vs theory, Z-score bar chart, p-value bar chart, and flagged-block table.
- Side-by-side comparison table with M4 across model type, supervision, training data, output, and evaluation metrics.

## Next Step

Write the final PDF report (2–3 pages) covering: cryptographic metrics and their meaning, M4 model justification and evaluation results, M7 model comparison, and external references.

## Main Problem or Blocker

No current blockers. Mempool.space `/v1/blocks` lacks `medianFee` for older blocks — resolved by falling back to `avgFeeRate`. Real-time mempool depth (WebSocket) would improve M4 accuracy but is not required.

## APIs Used

| API | Base URL | Used by |
|---|---|---|
| Blockstream | `blockstream.info/api` | M1, M2, M5, M7 |
| Mempool.space | `mempool.space/api` | M4 |
| Blockchain.info | `blockchain.info` | M3, M6 |

All APIs are free and require no registration or API key.

## Project Structure

```
blockchain-dashboard-foreverprogramming/
├── README.md
├── requirements.txt               ← requests, pandas, numpy, plotly, streamlit, scikit-learn, scipy
├── app.py                         ← Dashboard entry point (sidebar nav, global CSS)
├── api/
│   ├── __init__.py
│   └── blockchain_client.py       ← All API calls + realistic mock fallback data
├── modules/
│   ├── __init__.py
│   ├── m1_pow_monitor.py           ← PoW Monitor
│   ├── m2_block_header.py          ← Block Header Analyzer
│   ├── m3_difficulty_history.py    ← Difficulty History
│   ├── m4_ai_component.py          ← Fee Estimator (Gradient Boosting)
│   ├── m5_merkle_verifier.py       ← Merkle Proof Verifier
│   ├── m6_security_score.py        ← 51% Attack Cost + Nakamoto §11
│   └── m7_anomaly_detector.py      ← Block Anomaly Detector
└── report/
    └── CryptoChain_Report.pdf      ← Final project report
```

## Cryptographic Concepts Applied

| Concept | Where | Notes |
|---|---|---|
| SHA-256d | M2, M5 | `SHA256(SHA256(data))` — Bitcoin's standard double hash |
| Proof of Work | M1, M2 | Valid hash ≤ target; verified locally with `hashlib` |
| `bits` → target decoding | M1, M2, M6 | `target = coef × 2^(8×(exp−3))` |
| Difficulty formula | M1, M3 | `difficulty = genesis_target / current_target` |
| Hash rate estimate | M1, M6 | `hashrate ≈ difficulty × 2³² / 600` |
| Merkle tree | M2, M5 | SHA-256d binary tree; odd-node duplication; inclusion proof |
| Exponential inter-arrival | M1, M7 | Mining is a Bernoulli process → Exp(λ = 1/600) |
| Nakamoto §11 | M6 | Exact Poisson formula for 51% attack success probability |
| Difficulty adjustment | M3 | `new_d = old_d × actual_time / 1,209,600 s`, clamped to [¼, 4] |

## References

- Nakamoto, S. (2008). *Bitcoin: A Peer-to-Peer Electronic Cash System*. https://bitcoin.org/bitcoin.pdf
- Blockstream API documentation. https://github.com/Blockstream/esplora/blob/master/API.md
- Mempool.space API documentation. https://mempool.space/docs/api
- Blockchain.info Charts API. https://www.blockchain.com/explorer/charts

<!-- student-repo-auditor:teacher-feedback:start -->
## Teacher Feedback

### Kick-off Review

Review time: 2026-05-13 13:31 CEST
Status: Green

Strength:
- I can see the dashboard structure integrating the checkpoint modules.

Improve now:
- The README should now reflect the checkpoint more explicitly, including progress, blockers, and updated module status.

Next step:
- Update the README so progress, blockers, module status, and next step match the checkpoint format exactly.
<!-- student-repo-auditor:teacher-feedback:end -->
