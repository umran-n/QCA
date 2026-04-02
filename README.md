# QCA API
> Event-driven equity intelligence powered by QII

Determines whether an upcoming capital allocation event will generate
a CONSTRUCTIVE or DESTRUCTIVE equity signal - before price moves.

Built on the Quantum Capital Allocation framework (Papers I-IX, SSRN).

## Endpoints

| Family | Endpoint | Description |
|--------|----------|-------------|
| QII | GET /v1/qii/prescreen | Full QII regime signal |
| QII | GET /v1/qii/score | Lightweight QII score |
| Events | GET /v1/events/archetype | Seven-archetype classification |
| Coherence | GET /v1/coherence/tau | Signal holding window |
| Entanglement | GET /v1/entanglement/deff | Firm fragility score |

## Platform Architecture

- QCA API - the platform (this repo)
- QII - the flagship signal
- QCA Unified - coming (Product 3)

## Authentication

- `/health`, `/docs`, `/redoc`, and `/openapi.json` are public
- all `/v1/*` endpoints require upstream authentication
- direct access can use `X-API-Key` with `QCA_API_KEY`
- RapidAPI gateway access can use the hidden `X-RapidAPI-Proxy-Secret` with `RAPIDAPI_PROXY_SECRET`

## Research Pipeline

The closest-public replication pipeline remains available for offline runs:

```powershell
python scripts/run_qca_replication.py --refresh
```

Artifacts are written under `outputs/qca/latest/` and remain labeled `data_tier="public_proxy"`.
