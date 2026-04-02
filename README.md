# QCA Closest-Public Replication

This repository contains the closest-public replication pipeline for the Quantum Capital Allocation research program based on Papers VIII and IX.

The current implementation is explicit about data quality and labels every emitted artifact with `data_tier="public_proxy"`.

## Scope

- seven-firm QCA event-study sample
- `2019-01-01` to `2025-12-15` target window
- `2025-01-01` to `2025-12-15` out-of-sample holdout
- QII feature construction with public-proxy inputs
- coherence / `tau` module
- entanglement / `d_eff` screen
- contagion edge table

## Run

```powershell
python scripts/run_qca_replication.py --refresh
```

Outputs are written under `outputs/qca/latest/`.

## Tests

```powershell
python -m unittest tests.test_qca_math tests.test_qca_pipeline
```

## Important Note

This is a closest-public replication, not an institutional data build. The management sentiment path currently supports the documented public fallback and keeps the FinBERT upgrade path intact for a later higher-fidelity release.
