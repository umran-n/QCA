from __future__ import annotations

import math
import os
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .clients import fetch_sec_company_facts
from .config import load_json
from .discovery import apply_manual_overlay, build_event_ledger
from .features import (
    _amazon_internal_particle,
    _authorization_balance_before_event,
    _build_price_universe,
    _close_lookup,
    _dividend_yield,
    _management_score,
    _market_cap,
    _market_price_proxy,
    _options_realized_skew_proxy,
    _realized_wave,
    compute_feature_panel,
    compute_qii,
    compute_theta,
)
from .models import run_analytics
from .pipeline import _load_fixture_data, _resolve_config_paths
from .utils import closest_previous_date, format_date, parse_date, safe_div, stddev


PLATFORM = "QCA API"
SIGNAL_NAME = "QII"
TAGLINE = "Event-driven equity intelligence powered by QII"
API_VERSION = "1.0.0"
DATA_TIER = "public_proxy"
DIRECT_API_KEY_ENV = "QCA_API_KEY"
RAPID_PROXY_SECRET_ENV = "RAPIDAPI_PROXY_SECRET"
PAPER_URLS = [
    "https://github.com/umran-n/QCA/blob/main/paper/pdfs/Non-Linear%20Signal%20Extraction%20in%20Event-Driven%20Equities%20-%20A%20Unified%20Interference%20Functional%20%28QCA%20Series%20Paper%20IX%29.pdf",
    "https://github.com/umran-n/QCA/blob/main/paper/pdfs/Non-Stationary%20Valuation%20Dynamics%20in%20Mega-Cap%20Equities%20-%20A%20Unified%20Analysis%20of%20the%20Seven%20Archetypes%20%28Quantum%20Capital%20Allocation%20Series%20Paper%20VIII%20Meta-Synthesis%29.pdf",
]
H_T_V1 = 1.0  # v2: replace with transcript freshness decay


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _serialize_number(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _coverage_label(values: list[float | None]) -> str:
    present = sum(value is not None for value in values)
    if present == len(values):
        return "full"
    if present == 0:
        return "missing"
    return "partial"


def _regime_for_signal(qii_score: float, theta_degrees: float) -> str:
    if qii_score > 0.4 and theta_degrees < 60.0:
        return "CONSTRUCTIVE"
    if qii_score > 0.0 and theta_degrees < 90.0:
        return "CONSTRUCTIVE_MODERATE"
    if qii_score < 0.0 or theta_degrees > 120.0:
        return "DESTRUCTIVE"
    return "NEUTRAL"


def _conviction_for_signal(qii_score: float, theta_degrees: float) -> str:
    if qii_score > 0.5 and theta_degrees < 60.0:
        return "high"
    if qii_score > 0.2 and theta_degrees < 90.0:
        return "moderate"
    return "low"


def _qii_interpretation(regime: str) -> str:
    interpretations = {
        "CONSTRUCTIVE": "Constructive capital allocation signal. Mechanical capacity aligned with market expectations and management narrative.",
        "CONSTRUCTIVE_MODERATE": "Moderately constructive signal. Capital allocation capacity is positive, but alignment is less complete across the public proxy stack.",
        "DESTRUCTIVE": "Destructive capital allocation signal. Market expectations and phase alignment are working against the event setup.",
        "NEUTRAL": "Neutral capital allocation signal. Public-proxy evidence is mixed and does not resolve into a high-conviction QII view.",
    }
    return interpretations[regime]


def _entanglement_tier(d_eff: float) -> str:
    if d_eff < 2.0:
        return "low"
    if d_eff < 3.0:
        return "moderate"
    if d_eff < 3.5:
        return "elevated"
    return "high"


def _tier_description(tier: str) -> str:
    descriptions = {
        "low": "Low driver coupling. QII reads cleanly with limited cross-driver interference.",
        "moderate": "Moderate driver coupling. QII signal readable but watch for cross-driver amplification.",
        "elevated": "Elevated driver coupling. Cross-driver interactions can distort apparent event clarity.",
        "high": "High driver coupling. Tail-risk spillovers and contagion effects deserve active monitoring.",
    }
    return descriptions[tier]


def _regime_matrix_cell(theta_degrees: float, d_eff: float) -> str:
    if theta_degrees < 30.0:
        row = "A"
    elif theta_degrees < 60.0:
        row = "B"
    elif theta_degrees < 120.0:
        row = "C"
    else:
        row = "D"
    tier = _entanglement_tier(d_eff)
    column = {"low": 1, "moderate": 2, "elevated": 3, "high": 4}[tier]
    return f"{row}{column}"


def _driver_weights(particle_amplitude: float, wave_amplitude: float, theta_degrees: float) -> dict[str, float]:
    phase_weight = max(0.0, 1.0 - (theta_degrees / 180.0))
    raw = {
        "P_mechanical": abs(particle_amplitude),
        "W_volatility": abs(wave_amplitude),
        "theta_phase": phase_weight,
    }
    total = sum(raw.values())
    if total == 0:
        return {key: round(1.0 / len(raw), 4) for key in raw}
    return {key: round(value / total, 4) for key, value in raw.items()}


def _entropy_from_probabilities(probabilities: list[float]) -> tuple[float, float]:
    cleaned = [max(float(value), 0.0) for value in probabilities]
    total = sum(cleaned)
    if total <= 0:
        normalized = [1.0 / len(cleaned) for _ in cleaned] if cleaned else []
    else:
        normalized = [value / total for value in cleaned]
    entropy = -sum(value * math.log(value, 2) for value in normalized if value > 0)
    return entropy, 2.0 ** entropy


def _archetype_payload(feature_row: dict, d_eff: float, entropy_bits: float, theta_std: float) -> tuple[str, str, str]:
    particle_amplitude = float(feature_row["particle_amplitude"])
    wave_amplitude = float(feature_row["wave_amplitude"])
    theta_degrees = float(feature_row["theta_degrees"])
    qii_score = float(feature_row["qii"])

    if d_eff >= 3.5:
        return "entangled", "EN", "Driver interactions dominate. High effective dimensionality suggests the signal is coupled across multiple state variables."
    if theta_std >= 25.0 and entropy_bits >= 1.5:
        return "decoherent", "DC", "Phase behavior is unstable across the historical event set, which weakens clean signal persistence."
    if theta_degrees > 120.0 or qii_score < 0.0:
        return "destructive_interference", "DI", "Particle and wave terms are materially misaligned, producing destructive interference."
    if theta_degrees < 30.0 and qii_score > 0.0:
        return "constructive_interference", "CI", "Particle and wave terms are tightly aligned, producing constructive interference."
    scale = max(abs(particle_amplitude), abs(wave_amplitude), 1e-9)
    if abs(particle_amplitude - wave_amplitude) / scale <= 0.2 and theta_degrees < 45.0:
        return "phase_locked", "PL", "Particle and wave amplitudes are balanced with low phase slippage."
    if particle_amplitude >= wave_amplitude:
        return "particle_dominant", "PD", "Mechanical capital return capacity dominates. Firm has consistent authorization size relative to market cap."
    return "wave_dominant", "WD", "Volatility and expectation effects dominate the event setup more than mechanical capacity."


class QCAAPIError(Exception):
    def __init__(
        self,
        status_code: int,
        error: str,
        message: str,
        *,
        ticker: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error = error
        self.message = message
        self.ticker = ticker
        self.suggestion = suggestion or "QCA API covers large-cap equities with SEC EDGAR filings."


def _error_payload(error: str, message: str, *, ticker: str | None = None, suggestion: str | None = None) -> dict[str, Any]:
    timestamp = _utc_timestamp()
    payload = {
        "error": error,
        "message": message,
        "platform": PLATFORM,
        "data_tier": DATA_TIER,
        "suggestion": suggestion or "QCA API covers large-cap equities with SEC EDGAR filings.",
        "timestamp": timestamp,
        "computed_at": timestamp,
    }
    if ticker is not None:
        payload["ticker"] = ticker
    return payload


def _public_path(path: str) -> bool:
    return path in {"/health", "/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"}


def _authorized_request(request: Request) -> bool:
    direct_key = os.getenv(DIRECT_API_KEY_ENV, "").strip()
    proxy_secret = os.getenv(RAPID_PROXY_SECRET_ENV, "").strip()
    request_api_key = request.headers.get("X-API-Key", "").strip()
    request_proxy_secret = request.headers.get("X-RapidAPI-Proxy-Secret", "").strip()
    return bool((direct_key and request_api_key == direct_key) or (proxy_secret and request_proxy_secret == proxy_secret))


def _unauthorized_response() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content=_error_payload(
            "unauthorized",
            "Valid X-API-Key required.",
            suggestion="Use the RapidAPI gateway or provide a valid X-API-Key for direct access.",
        ),
    )


class QCAService:
    def __init__(self, root: Path, config_path: Path) -> None:
        self.root = root
        self.config_path = config_path
        self.config = _resolve_config_paths(root, load_json(config_path))
        self.fixture_data = _load_fixture_data(self.config)
        self.raw_dir = root / "data" / "qca" / "raw"
        self._lock = Lock()
        self._baseline_cache: dict[str, Any] | None = None
        self._company_facts_cache: dict[str, dict] = {}
        self._price_cache: dict[str, tuple[dict[str, dict], dict[str, tuple[dict[date, float], list[date], dict[date, float]]]]] = {}
        self._firm_lookup = {firm["ticker"]: firm for firm in self.config["firms"]}

    def supported_tickers(self) -> set[str]:
        return set(self._firm_lookup)

    def firm_for(self, ticker: str) -> dict:
        normalized = ticker.upper()
        if normalized not in self._firm_lookup:
            raise QCAAPIError(
                404,
                "ticker_not_found",
                f"No public data available for ticker: {normalized}",
                ticker=normalized,
            )
        return self._firm_lookup[normalized]

    def baseline(self, refresh: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._baseline_cache is not None and not refresh:
                return self._baseline_cache
            if self.fixture_data and self.config.get("fixtures", {}).get("disable_discovery"):
                ledger, gap_rows = apply_manual_overlay([], self.config)
            else:
                ledger, gap_rows = build_event_ledger(self.config, self.raw_dir, refresh=refresh)
            updated_ledger, feature_rows, return_rows = compute_feature_panel(
                ledger,
                self.config,
                self.raw_dir,
                refresh=refresh,
                fixture_data=self.fixture_data,
            )
            analytics = run_analytics(
                feature_rows,
                return_rows,
                self.config,
                self.raw_dir,
                refresh=refresh,
                fixture_data=self.fixture_data,
            )
            self._baseline_cache = {
                "ledger": updated_ledger,
                "gap_rows": gap_rows,
                "feature_rows": feature_rows,
                "return_rows": return_rows,
                "analytics": analytics,
            }
            return self._baseline_cache

    def company_facts(self, ticker: str, refresh: bool = False) -> dict:
        normalized = ticker.upper()
        with self._lock:
            if normalized in self._company_facts_cache and not refresh:
                return self._company_facts_cache[normalized]
        if self.fixture_data and normalized in self.fixture_data.get("company_facts", {}):
            payload = self.fixture_data["company_facts"][normalized]
        else:
            firm = self.firm_for(normalized)
            payload = fetch_sec_company_facts(
                firm["cik"],
                self.raw_dir,
                self.config["sec"]["user_agent"],
                refresh=refresh,
            )
        with self._lock:
            self._company_facts_cache[normalized] = payload
        return payload

    def price_bundle(
        self,
        event_day: date,
        refresh: bool = False,
    ) -> tuple[dict[str, dict], dict[str, tuple[dict[date, float], list[date], dict[date, float]]]]:
        sample_end = max(parse_date(self.config["sample"]["end"]), event_day)
        cache_key = format_date(sample_end)
        with self._lock:
            if cache_key in self._price_cache and not refresh:
                return self._price_cache[cache_key]
        config = deepcopy(self.config)
        config["sample"]["end"] = format_date(sample_end)
        payloads = _build_price_universe(config, self.raw_dir, refresh=refresh, fixture_data=self.fixture_data)
        price_maps = {symbol: _close_lookup(payload) for symbol, payload in payloads.items()}
        with self._lock:
            self._price_cache[cache_key] = (payloads, price_maps)
        return payloads, price_maps

    def prescreen_snapshot(self, ticker: str, event_day: date, refresh: bool = False) -> dict[str, Any]:
        normalized = ticker.upper()
        firm = self.firm_for(normalized)
        baseline = self.baseline(refresh=refresh)
        company_facts = self.company_facts(normalized, refresh=refresh)
        price_payloads, price_maps = self.price_bundle(event_day, refresh=refresh)

        market_symbol = self.config["market_data"]["market_symbol"]
        sector_symbol = self.config["market_data"]["sector_symbol"]
        stock_closes, stock_dates, stock_returns = price_maps[normalized]
        _, spy_dates, _ = price_maps[market_symbol]
        sector_closes, _, _ = price_maps[sector_symbol]

        asof = closest_previous_date(spy_dates, event_day - timedelta(days=1))
        if asof is None:
            raise QCAAPIError(
                500,
                "compute_error",
                f"Unable to locate a pre-event trading day for {normalized} on {format_date(event_day)}.",
                ticker=normalized,
            )

        market_cap = _market_cap(company_facts, stock_closes, asof)
        close_day = closest_previous_date(stock_dates, asof)
        close_value = stock_closes.get(close_day) if close_day else None
        if market_cap is None or close_value is None:
            raise QCAAPIError(
                500,
                "compute_error",
                f"Insufficient public market-cap inputs to compute QII for {normalized}.",
                ticker=normalized,
            )

        dividend_yield = _dividend_yield(price_payloads[normalized], asof, close_value)
        remaining_authorization = _authorization_balance_before_event(
            baseline["ledger"],
            normalized,
            event_day,
            company_facts,
            parse_date(self.config["sample"]["start"]),
        )
        if firm["particle_variant"] == "internal_reallocation":
            particle_amplitude = _amazon_internal_particle(company_facts, market_cap, asof)
        else:
            particle_amplitude = safe_div(remaining_authorization, market_cap) + dividend_yield

        wave_amplitude = _realized_wave(stock_returns, stock_dates, asof)
        s_market = _market_price_proxy(
            stock_closes,
            stock_dates,
            sector_closes,
            asof,
            int(self.config["market_sentiment"]["price_proxy_window_days"]),
            float(self.config["market_sentiment"]["scale"]),
        )
        s_options = _options_realized_skew_proxy(
            stock_returns,
            stock_dates,
            asof,
            int(self.config["options_sentiment"]["realized_skew_window_days"]),
        )
        s_mgmt, management_text_source, management_model, management_date = _management_score(
            firm,
            asof,
            self.config,
            self.raw_dir,
            refresh=refresh,
            fixture_data=self.fixture_data,
        )

        theta_degrees = compute_theta(s_mgmt, s_market, s_options, h_t=H_T_V1)
        qii_score = compute_qii(particle_amplitude, wave_amplitude, theta_degrees)
        coverage = _coverage_label([s_mgmt, s_market, s_options])
        if qii_score is None or theta_degrees is None or coverage != "full":
            raise QCAAPIError(
                500,
                "compute_error",
                f"Insufficient labeled public-proxy inputs to compute QII for {normalized}.",
                ticker=normalized,
                suggestion="Confirm the ticker is in the QCA universe and that the relevant public filings and price history are available.",
            )

        regime = _regime_for_signal(qii_score, theta_degrees)
        conviction = _conviction_for_signal(qii_score, theta_degrees)
        watch_flags: list[str] = []
        if theta_degrees > 90.0:
            watch_flags.append("phase_misalignment")
        if qii_score < 0.0:
            watch_flags.append("negative_interference")

        tau_base_days = int(round(self.tau_snapshot(normalized, theta_degrees)["tau_days"]))
        return {
            "ticker": normalized,
            "event_date": format_date(event_day),
            "t0_date": format_date(event_day),
            "signal": {
                "regime": regime,
                "qii_score": _serialize_number(qii_score, digits=4),
                "qii_interpretation": _qii_interpretation(regime),
                "conviction": conviction,
                "watch_flags": watch_flags,
            },
            "components": {
                "P": _serialize_number(particle_amplitude, digits=4),
                "P_source": "sec_edgar_public_proxy",
                "W": _serialize_number(wave_amplitude, digits=4),
                "W_source": "realized_vol_proxy",
                "theta_degrees": _serialize_number(theta_degrees, digits=2),
                "S_mgmt": _serialize_number(s_mgmt, digits=4),
                "S_mgmt_source": management_model,
                "S_mgmt_text_source": management_text_source,
                "S_mgmt_text_date": format_date(management_date) if management_date else None,
                "S_market": _serialize_number(s_market, digits=4),
                "S_market_source": "market_price_proxy",
                "S_options": _serialize_number(s_options, digits=4),
                "S_options_source": "options_realized_skew_proxy",
                "h_t": H_T_V1,
                "theta_component_coverage": coverage,
                "theta_source_variant": "faithful_proxy_bundle",
            },
            "coherence_preview": {
                "tau_base_days": tau_base_days,
                "note": "Full coherence analysis at /v1/coherence/tau",
            },
            "platform": PLATFORM,
            "signal_name": SIGNAL_NAME,
            "data_tier": DATA_TIER,
            "data_tier_note": "All components from public sources. Institutional upgrade replaces inputs only - formulas unchanged.",
            "papers": PAPER_URLS,
            "computed_at": _utc_timestamp(),
        }

    def latest_feature_row(self, ticker: str) -> dict[str, Any]:
        normalized = ticker.upper()
        self.firm_for(normalized)
        rows = [row for row in self.baseline()["feature_rows"] if row["ticker"] == normalized and row.get("qii") is not None]
        if not rows:
            raise QCAAPIError(
                404,
                "ticker_not_found",
                f"No historical QCA feature panel available for ticker: {normalized}",
                ticker=normalized,
            )
        rows.sort(key=lambda item: item["announcement_date"])
        return rows[-1]

    def _fallback_entanglement_snapshot(self, ticker: str) -> dict[str, Any]:
        feature_row = self.latest_feature_row(ticker)
        driver_weights = _driver_weights(
            float(feature_row["particle_amplitude"] or 0.0),
            float(feature_row["wave_amplitude"] or 0.0),
            float(feature_row["theta_degrees"] or 90.0),
        )
        eigenvalues = list(driver_weights.values())
        entropy_bits, d_eff = _entropy_from_probabilities(eigenvalues)
        tier = _entanglement_tier(d_eff)
        return {
            "ticker": ticker.upper(),
            "d_eff": _serialize_number(d_eff, digits=4),
            "entropy_S": _serialize_number(entropy_bits, digits=4),
            "eigenvalues": [_serialize_number(value, digits=4) for value in eigenvalues],
            "entanglement_tier": tier,
            "tier_description": _tier_description(tier),
            "tail_risk_flag": d_eff >= 3.5,
            "source_variant": "driver_weight_fallback",
            "platform": PLATFORM,
            "data_tier": DATA_TIER,
            "computed_at": _utc_timestamp(),
        }

    def entanglement_snapshot(self, ticker: str) -> dict[str, Any]:
        normalized = ticker.upper()
        self.firm_for(normalized)
        row = next(
            (item for item in self.baseline()["analytics"]["entanglement_rows"] if item["ticker"] == normalized and item.get("d_eff") is not None),
            None,
        )
        if row is None:
            return self._fallback_entanglement_snapshot(normalized)
        d_eff = float(row["d_eff"])
        tier = _entanglement_tier(d_eff)
        tail_risk_flag = d_eff >= 3.5
        return {
            "ticker": normalized,
            "d_eff": _serialize_number(d_eff, digits=4),
            "entropy_S": _serialize_number(float(row["entropy_bits"]), digits=4),
            "eigenvalues": [_serialize_number(float(value), digits=4) for value in row.get("eigenvalues", [])],
            "entanglement_tier": tier,
            "tier_description": _tier_description(tier),
            "tail_risk_flag": tail_risk_flag,
            "source_variant": "historical_driver_system",
            "platform": PLATFORM,
            "data_tier": DATA_TIER,
            "computed_at": _utc_timestamp(),
        }

    def archetype_snapshot(self, ticker: str) -> dict[str, Any]:
        feature_row = self.latest_feature_row(ticker)
        entanglement = self.entanglement_snapshot(ticker)
        baseline_rows = [
            row for row in self.baseline()["feature_rows"] if row["ticker"] == ticker.upper() and row.get("theta_degrees") is not None
        ]
        theta_std = stddev([float(row["theta_degrees"]) for row in baseline_rows])
        d_eff = float(entanglement["d_eff"])
        entropy_bits = float(entanglement["entropy_S"])
        archetype, code, description = _archetype_payload(feature_row, d_eff, entropy_bits, theta_std)
        return {
            "ticker": ticker.upper(),
            "archetype": archetype,
            "archetype_code": code,
            "description": description,
            "d_eff": entanglement["d_eff"],
            "entropy_S": entanglement["entropy_S"],
            "entanglement_tier": entanglement["entanglement_tier"],
            "driver_weights": _driver_weights(
                float(feature_row["particle_amplitude"]),
                float(feature_row["wave_amplitude"]),
                float(feature_row["theta_degrees"]),
            ),
            "regime_matrix_cell": _regime_matrix_cell(float(feature_row["theta_degrees"]), d_eff),
            "platform": PLATFORM,
            "data_tier": DATA_TIER,
            "computed_at": _utc_timestamp(),
        }

    def tau_snapshot(self, ticker: str, theta_degrees: float) -> dict[str, Any]:
        normalized = ticker.upper()
        self.firm_for(normalized)
        analytics = self.baseline()["analytics"]
        decay_law = analytics["decay_law"]
        tau_max = float(decay_law["tau_max"]) if decay_law.get("tau_max") is not None else 180.0
        alpha = float(decay_law["alpha"]) if decay_law.get("alpha") is not None else 0.035
        tau_days = tau_max * math.exp(-alpha * theta_degrees)
        ticker_rows = [
            row
            for row in analytics["coherence_rows"]
            if row["ticker"] == normalized and row.get("qualified") and row.get("fit_r2") is not None
        ]
        r_squared_fit = max((float(row["fit_r2"]) for row in ticker_rows), default=float(decay_law.get("r2") or 0.84))
        return {
            "ticker": normalized,
            "theta_degrees": _serialize_number(theta_degrees, digits=2),
            "tau_days": _serialize_number(tau_days, digits=2),
            "tau_formula": "tau_max * exp(-alpha * theta)",
            "tau_max": _serialize_number(tau_max, digits=2),
            "alpha": _serialize_number(alpha, digits=4),
            "r_squared_fit": _serialize_number(r_squared_fit, digits=2),
            "interpretation": f"QII signal coherence window: ~{int(round(tau_days))} trading days post-announcement.",
            "regime_adjusted_note": "Macro-adjusted tau available in QCA Unified (Product 3).",
            "platform": PLATFORM,
            "data_tier": DATA_TIER,
            "computed_at": _utc_timestamp(),
        }


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config_path(root: Path) -> Path:
    configured = os.getenv("QCA_CONFIG_PATH")
    if configured:
        return Path(configured)
    return root / "config" / "qca_config.json"


def create_app(root: Path | None = None, config_path: Path | None = None) -> FastAPI:
    repo_root = root or _default_root()
    resolved_config_path = config_path or _default_config_path(repo_root)
    service = QCAService(repo_root, resolved_config_path)
    app = FastAPI(
        title=PLATFORM,
        description="Event-driven equity intelligence powered by QII. Based on the Quantum Capital Allocation framework (Papers I-IX).",
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.qca_service = service

    @app.middleware("http")
    async def enforce_api_auth(request: Request, call_next):
        path = request.url.path
        if _public_path(path) or not path.startswith("/v1/"):
            return await call_next(request)
        if not _authorized_request(request):
            return _unauthorized_response()
        return await call_next(request)

    @app.exception_handler(QCAAPIError)
    async def qca_api_error_handler(_: Request, exc: QCAAPIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                exc.error,
                exc.message,
                ticker=exc.ticker,
                suggestion=exc.suggestion,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        first_error = exc.errors()[0] if exc.errors() else {"msg": "Invalid request parameters."}
        message = first_error.get("msg", "Invalid request parameters.")
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                "bad_params",
                message,
                suggestion="Check ticker, date, and numeric query parameters before retrying the request.",
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                "compute_error",
                str(exc) or "Unexpected QCA compute error.",
                suggestion="Retry once. If the error persists, refresh the public data cache and inspect the service logs.",
            ),
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        timestamp = _utc_timestamp()
        return {
            "status": "ok",
            "service": PLATFORM,
            "platform": PLATFORM,
            "tagline": TAGLINE,
            "version": API_VERSION,
            "data_tier": DATA_TIER,
            "endpoints": {
                "qii": "/v1/qii",
                "events": "/v1/events",
                "coherence": "/v1/coherence",
                "entanglement": "/v1/entanglement",
            },
            "timestamp": timestamp,
            "computed_at": timestamp,
        }

    @app.get("/v1/qii/prescreen")
    def qii_prescreen(
        request: Request,
        ticker: str = Query(..., min_length=1),
        event_date: date = Query(...),
        refresh: bool = Query(False),
    ) -> dict[str, Any]:
        qca_service: QCAService = request.app.state.qca_service
        return qca_service.prescreen_snapshot(ticker, event_date, refresh=refresh)

    @app.get("/v1/qii/score")
    def qii_score(
        request: Request,
        ticker: str = Query(..., min_length=1),
        event_date: date = Query(...),
        refresh: bool = Query(False),
    ) -> dict[str, Any]:
        qca_service: QCAService = request.app.state.qca_service
        prescreen = qca_service.prescreen_snapshot(ticker, event_date, refresh=refresh)
        return {
            "ticker": prescreen["ticker"],
            "event_date": prescreen["event_date"],
            "qii_score": prescreen["signal"]["qii_score"],
            "regime": prescreen["signal"]["regime"],
            "conviction": prescreen["signal"]["conviction"],
            "signal_name": SIGNAL_NAME,
            "platform": PLATFORM,
            "data_tier": DATA_TIER,
            "computed_at": _utc_timestamp(),
        }

    @app.get("/v1/events/archetype")
    def event_archetype(
        request: Request,
        ticker: str = Query(..., min_length=1),
    ) -> dict[str, Any]:
        qca_service: QCAService = request.app.state.qca_service
        return qca_service.archetype_snapshot(ticker)

    @app.get("/v1/coherence/tau")
    def coherence_tau(
        request: Request,
        ticker: str = Query(..., min_length=1),
        theta: float = Query(..., ge=0.0, le=180.0),
    ) -> dict[str, Any]:
        qca_service: QCAService = request.app.state.qca_service
        return qca_service.tau_snapshot(ticker, theta)

    @app.get("/v1/entanglement/deff")
    def entanglement_deff(
        request: Request,
        ticker: str = Query(..., min_length=1),
    ) -> dict[str, Any]:
        qca_service: QCAService = request.app.state.qca_service
        return qca_service.entanglement_snapshot(ticker)

    return app


app = create_app()
