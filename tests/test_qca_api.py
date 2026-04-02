from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qca_replication.api import create_app
from tests.test_qca_pipeline import _fixture_bundle, _test_config


class QCAApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._prior_env = {
            "QCA_API_KEY": os.environ.get("QCA_API_KEY"),
            "RAPIDAPI_PROXY_SECRET": os.environ.get("RAPIDAPI_PROXY_SECRET"),
        }
        os.environ["QCA_API_KEY"] = "test-direct-key"
        os.environ["RAPIDAPI_PROXY_SECRET"] = "test-proxy-secret"
        self.tmpdir = Path(tempfile.mkdtemp(dir=ROOT))
        self.bundle_path = self.tmpdir / "fixture_bundle.json"
        self.overlay_path = self.tmpdir / "manual_overlay.json"
        self.output_dir = self.tmpdir / "outputs"
        self.config_path = self.tmpdir / "qca_test_config.json"
        self.bundle_path.write_text(json.dumps(_fixture_bundle()), encoding="utf-8")
        self.overlay_path.write_text(
            json.dumps(
                {
                    "manual_events": [
                        {
                            "ticker": "AAPL",
                            "announcement_date": "2024-05-02",
                            "event_family": "dual_announcement",
                            "event_title": "Apple spring authorization",
                            "authorization_amount": 110000000000,
                            "dividend_per_share": 0.25,
                            "manual_source_urls": ["fixture://aapl/20240502"],
                        },
                        {
                            "ticker": "AAPL",
                            "announcement_date": "2025-01-30",
                            "event_family": "dual_announcement",
                            "event_title": "Apple winter authorization",
                            "authorization_amount": 90000000000,
                            "dividend_per_share": 0.26,
                            "manual_source_urls": ["fixture://aapl/20250130"],
                        },
                        {
                            "ticker": "AMZN",
                            "announcement_date": "2024-08-01",
                            "event_family": "internal_reallocation",
                            "event_title": "Amazon AI reallocation",
                            "manual_source_urls": ["fixture://amzn/20240801"],
                        },
                        {
                            "ticker": "AMZN",
                            "announcement_date": "2025-01-30",
                            "event_family": "internal_reallocation",
                            "event_title": "Amazon AI reallocation follow-up",
                            "manual_source_urls": ["fixture://amzn/20250130"],
                        },
                    ],
                    "exclusions": [],
                }
            ),
            encoding="utf-8",
        )
        self.config_path.write_text(
            json.dumps(_test_config(self.tmpdir, self.bundle_path, self.overlay_path, self.output_dir), indent=2),
            encoding="utf-8",
        )
        self.app = create_app(root=ROOT, config_path=self.config_path)
        self.client = TestClient(self.app)
        self.auth_headers = {"X-API-Key": "test-direct-key"}
        self.proxy_headers = {"X-RapidAPI-Proxy-Secret": "test-proxy-secret"}

    def tearDown(self) -> None:
        self.client.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        for key, value in self._prior_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_api_endpoints_include_platform_metadata(self) -> None:
        endpoints = [
            "/health",
            "/v1/qii/prescreen?ticker=AAPL&event_date=2025-01-30",
            "/v1/qii/score?ticker=AAPL&event_date=2025-01-30",
            "/v1/events/archetype?ticker=AAPL",
            "/v1/coherence/tau?ticker=AAPL&theta=52",
            "/v1/coherence/window?ticker=AAPL&event_date=2025-01-30",
            "/v1/entanglement/deff?ticker=AAPL",
        ]
        for path in endpoints:
            headers = None if path == "/health" else self.auth_headers
            response = self.client.get(path, headers=headers)
            self.assertEqual(response.status_code, 200, msg=path)
            payload = response.json()
            self.assertEqual(payload["platform"], "QCA API")
            self.assertEqual(payload["data_tier"], "public_proxy")
            self.assertIn("computed_at", payload)

    def test_invalid_ticker_returns_structured_404(self) -> None:
        response = self.client.get("/v1/qii/score?ticker=XYZ&event_date=2025-01-30", headers=self.auth_headers)
        self.assertEqual(response.status_code, 404)
        payload = response.json()
        self.assertEqual(payload["error"], "ticker_not_found")
        self.assertEqual(payload["ticker"], "XYZ")
        self.assertEqual(payload["platform"], "QCA API")
        self.assertEqual(payload["data_tier"], "public_proxy")
        self.assertIn("timestamp", payload)

    def test_protected_endpoint_requires_auth(self) -> None:
        response = self.client.get("/v1/entanglement/deff?ticker=AAPL")
        self.assertEqual(response.status_code, 401)
        payload = response.json()
        self.assertEqual(payload["error"], "unauthorized")
        self.assertEqual(payload["platform"], "QCA API")
        self.assertEqual(payload["data_tier"], "public_proxy")
        self.assertIn("computed_at", payload)

    def test_rapid_proxy_secret_auth_works(self) -> None:
        response = self.client.get("/v1/entanglement/deff?ticker=AAPL", headers=self.proxy_headers)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["ticker"], "AAPL")
        self.assertEqual(payload["platform"], "QCA API")

    def test_entanglement_falls_back_when_historical_profile_missing(self) -> None:
        service = self.app.state.qca_service
        baseline = service.baseline()
        baseline["analytics"]["entanglement_rows"] = []
        response = self.client.get("/v1/entanglement/deff?ticker=AAPL", headers=self.auth_headers)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["ticker"], "AAPL")
        self.assertEqual(payload["source_variant"], "driver_weight_fallback")
        self.assertEqual(payload["platform"], "QCA API")

    def test_coherence_window_derives_theta_from_prescreen(self) -> None:
        prescreen = self.client.get("/v1/qii/prescreen?ticker=AAPL&event_date=2025-01-30", headers=self.auth_headers)
        self.assertEqual(prescreen.status_code, 200)
        prescreen_payload = prescreen.json()
        response = self.client.get("/v1/coherence/window?ticker=AAPL&event_date=2025-01-30", headers=self.auth_headers)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["ticker"], "AAPL")
        self.assertEqual(payload["event_date"], "2025-01-30")
        self.assertEqual(payload["signal_name"], "QII")
        self.assertEqual(payload["source_variant"], "derived_from_qii_prescreen")
        self.assertEqual(payload["theta_degrees"], prescreen_payload["components"]["theta_degrees"])
        self.assertEqual(payload["qii_score"], prescreen_payload["signal"]["qii_score"])


if __name__ == "__main__":
    unittest.main()
