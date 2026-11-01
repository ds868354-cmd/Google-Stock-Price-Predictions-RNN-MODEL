"""Tests generated with this bundle -- ci.yml runs `pytest tests/`.

They need no network and no real secrets: every test builds a throw-away api_keys.json with create_api_key.py and drives app.py
through FastAPI's TestClient. What is covered: /health, API-key enforcement (401 / revoked key), the 429 rate limit and -- for
tabular / ANN bundles -- a complete row scoring 200 with an integer class label, null values imputed with a warning, and an
incomplete row refused with 422 (never silently imputed).
"""
import importlib
import os
import sys

import pytest

BUNDLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BUNDLE_DIR)

from fastapi.testclient import TestClient  # noqa: E402

import create_api_key  # noqa: E402

MODEL_TYPE = "RNN"
ENDPOINTS = {"TABULAR_ML": "/predict/tabular", "ANN": "/predict/tabular", "CNN": "/predict/image", "RNN": "/predict/sequence",
             "NLP": "/predict/text", "RAG_LLM": "/query", "UNSUPERVISED_ML": "/predict/cluster",
             "ANOMALY_DETECTION": "/predict/anomaly", "TABULAR_RL": "/predict/action", "LLM_FINETUNE": "/predict/text"}
ENDPOINT = ENDPOINTS.get(MODEL_TYPE, "/predict/tabular")
LOADS_ASYNC = MODEL_TYPE in ("LLM_FINETUNE", "RAG_LLM")   # adapter / embedding model load in a background thread (may need a download)
TABULAR = MODEL_TYPE in ("TABULAR_ML", "ANN")
_ENV = ("REQUIRE_AUTH", "API_KEYS_PATH", "RATE_LIMIT_PER_MINUTE", "ALLOW_MISSING_FEATURES", "DECISION_THRESHOLD", "APP_ENV",
        "ENABLE_DOCS", "MAX_BODY_BYTES", "MAX_IMAGE_BYTES", "MAX_TEXT_CHARS", "PREDICT_ROLES", "TRUSTED_PROXY_HOPS", "HEALTH_VERBOSE", "AUTH_FAIL_LIMIT_PER_MINUTE",
        "API_KEYS_JSON", "DRIFT_WINDOW_SIZE", "DRIFT_MIN_WINDOW", "DRIFT_EVAL_EVERY", "DRIFT_PSI_THRESHOLD", "LOG_LATENCY")


def load_app(monkeypatch, **env):
    """Fresh import of app.py under a controlled environment (it reads its settings at import time)."""
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def keyed_client(monkeypatch, tmp_path, **env):
    keys = str(tmp_path / "api_keys.json")
    raw_key = create_api_key.create_key(keys, owner="pytest")
    module = load_app(monkeypatch, REQUIRE_AUTH="true", API_KEYS_PATH=keys, APP_ENV="production", **env)
    return TestClient(module.app), raw_key, keys


def test_health_reports_the_model_type_and_load_state(monkeypatch):
    module = load_app(monkeypatch, REQUIRE_AUTH="false", APP_ENV="staging")
    response = TestClient(module.app).get("/health")
    body = response.json()
    assert body.get("model_type") == MODEL_TYPE, body
    if LOADS_ASYNC:
        assert response.status_code in (200, 503), response.text
    else:
        assert response.status_code == 200, response.text
        assert body.get("model_loaded", body.get("onnx_loaded", True)) is not False, body
        assert not body.get("artifact_load_errors"), body


def test_predict_requires_a_valid_api_key(monkeypatch, tmp_path):
    client, raw_key, _ = keyed_client(monkeypatch, tmp_path)
    assert client.post(ENDPOINT, json={}).status_code == 401
    assert client.post(ENDPOINT, json={}, headers={"X-API-Key": "nope.nope"}).status_code == 401
    accepted = client.post(ENDPOINT, json={}, headers={"X-API-Key": raw_key})
    assert accepted.status_code not in (401, 403), accepted.text   # an empty body is a 422 -- the point is that auth passed
    assert client.get("/health").status_code in (200, 503)           # probes never need a key


def test_revoked_key_is_refused(monkeypatch, tmp_path):
    client, raw_key, keys = keyed_client(monkeypatch, tmp_path)
    assert client.post(ENDPOINT, json={}, headers={"X-API-Key": raw_key}).status_code not in (401, 403)
    create_api_key.revoke_key(keys, raw_key.split(".", 1)[0])
    assert client.post(ENDPOINT, json={}, headers={"X-API-Key": raw_key}).status_code == 401


def test_rate_limit_returns_429(monkeypatch, tmp_path):
    client, raw_key, _ = keyed_client(monkeypatch, tmp_path, RATE_LIMIT_PER_MINUTE="2")
    codes = [client.post(ENDPOINT, json={}, headers={"X-API-Key": raw_key}).status_code for _ in range(4)]
    assert codes[0] != 429 and 429 in codes, codes


def _sample_features(recipe):
    feats = {}
    medians = recipe.get("impute_medians") or []
    for i, col in enumerate(recipe.get("numeric_cols") or []):
        feats[col] = medians[i] if i < len(medians) else 0.0
    cats = ((recipe.get("encode_recipe") or {}).get("encoders") or {}).get("onehot_categories") or []
    for col, values in zip(recipe.get("categorical_cols") or [], cats):
        feats[col] = values[0] if values else ""
    return feats


def test_tabular_scoring_contract(monkeypatch):
    if not TABULAR:
        pytest.skip("tabular / ANN bundles only")
    module = load_app(monkeypatch, REQUIRE_AUTH="false", APP_ENV="staging", RATE_LIMIT_PER_MINUTE="0")
    if getattr(module, "_recipe", None) is None:
        pytest.skip("this bundle has no feature_recipe.json")
    client = TestClient(module.app)
    if client.get("/health").status_code != 200:
        pytest.skip("model artifact not loaded in this environment")
    feats = _sample_features(module._recipe)
    ok = client.post("/predict/tabular", json={"features": feats})
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert "prediction" in body, body
    if "probabilities" in body:
        # a whole-number class label must come back as an int (1), not a float (1.0)
        assert not (isinstance(body["prediction"], float) and body["prediction"].is_integer()), body
    numeric = list(module._recipe.get("numeric_cols") or [])
    if numeric:
        with_null = dict(feats)
        with_null[numeric[0]] = None           # explicit null = unknown: imputed AND reported back
        nulled = client.post("/predict/tabular", json={"features": with_null})
        assert nulled.status_code == 200, nulled.text
        assert nulled.json().get("warnings"), nulled.json()
    if len(feats) > 1:
        partial = client.post("/predict/tabular", json={"features": dict(list(feats.items())[:1])})
        assert partial.status_code == 422, partial.text   # omitted features are refused, never silently imputed


def _raw_example(recipe):
    """The RAW-input contract: engineered columns (date parts, geo distance, frequency codes) are replaced by the raw columns the
    server derives them from -- the same example the README shows."""
    feats = _sample_features(recipe)
    for spec in recipe.get("derived_features") or []:
        kind = spec.get("type")
        if kind == "date_parts":
            for part in spec.get("parts", []):
                feats.pop(spec["prefix"] + "_" + part, None)
            if "dob" in str(spec["source"]).lower() or "birth" in str(spec["source"]).lower():
                feats[spec["source"]] = "1985-03-20"
            else:
                feats[spec["source"]] = "2020-06-15 12:30:00" if "hour" in spec.get("parts", []) else "2020-06-15"
        elif kind == "geo_distance":
            feats.pop(spec["out"], None)
            for a, b in (("lat1", "lat2"), ("lon1", "lon2")):
                feats.setdefault(spec[a], 0.0)
                feats.setdefault(spec[b], feats[spec[a]])
        elif kind == "frequency":
            feats.pop(spec["out"], None)
            feats[spec["source"]] = next(iter(spec.get("map") or {}), "x")
        elif kind in ("ordinal", "category_map"):
            feats.pop(spec["out"], None)
            feats[spec["source"]] = next(iter(spec.get("map") or {}), "x")
        elif kind == "log1p":
            feats.pop(spec["out"], None)
    return feats


def _ready_tabular_client(monkeypatch, **env):
    if not TABULAR:
        pytest.skip("tabular / ANN bundles only")
    module = load_app(monkeypatch, REQUIRE_AUTH="false", APP_ENV="staging", RATE_LIMIT_PER_MINUTE="0", **env)
    if getattr(module, "_recipe", None) is None:
        pytest.skip("this bundle has no feature_recipe.json")
    client = TestClient(module.app)
    if client.get("/health").status_code != 200:
        pytest.skip("model artifact not loaded in this environment")
    return module, client


def test_tabular_raw_input_contract(monkeypatch):
    module, client = _ready_tabular_client(monkeypatch)
    if not module._recipe.get("derived_features"):
        pytest.skip("this bundle has no engineered columns")
    ok = client.post("/predict/tabular", json={"features": _raw_example(module._recipe)})
    assert ok.status_code == 200, ok.text


def test_tabular_non_finite_numbers(monkeypatch):
    module, client = _ready_tabular_client(monkeypatch)
    numeric = list(module._recipe.get("numeric_cols") or [])
    if not numeric:
        pytest.skip("no numeric features")
    feats = _sample_features(module._recipe)
    feats[numeric[0]] = "inf"
    assert client.post("/predict/tabular", json={"features": feats}).status_code == 422
    feats[numeric[0]] = "nan"                     # the text "nan" means unknown, exactly like null: imputed AND reported
    nan_resp = client.post("/predict/tabular", json={"features": feats})
    assert nan_resp.status_code == 200 and nan_resp.json().get("warnings"), nan_resp.text


def test_binary_classifier_without_threshold_is_not_ready(monkeypatch):
    module, client = _ready_tabular_client(monkeypatch)
    if not module._is_binary_classifier():
        pytest.skip("not a binary classifier")
    module._recipe.pop("decision_threshold", None)
    module._recipe.pop("ann_decision_threshold", None)
    assert client.get("/health").status_code == 503          # no 0.50 default: the pod must not look ready
    monkeypatch.setenv("DECISION_THRESHOLD", "0.4")           # a manual deploy-time value makes it ready again
    assert client.get("/health").status_code == 200
    ok = client.post("/predict/tabular", json={"features": _sample_features(module._recipe)})
    assert ok.status_code == 200, ok.text


def test_docs_are_closed_in_production(monkeypatch):
    module = load_app(monkeypatch, REQUIRE_AUTH="false", APP_ENV="production")
    client = TestClient(module.app)
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404
    module = load_app(monkeypatch, REQUIRE_AUTH="false", APP_ENV="production", ENABLE_DOCS="true")
    assert TestClient(module.app).get("/openapi.json").status_code == 200


def test_bad_env_values_do_not_crash_the_import(monkeypatch):
    module = load_app(monkeypatch, REQUIRE_AUTH="false", APP_ENV="staging", RATE_LIMIT_PER_MINUTE="sixty")
    assert module.RATE_LIMIT_PER_MINUTE == 60


def test_oversized_body_is_refused(monkeypatch):
    module = load_app(monkeypatch, REQUIRE_AUTH="false", APP_ENV="staging", MAX_BODY_BYTES="64")
    response = TestClient(module.app).post(ENDPOINT, content=b"x" * 500, headers={"Content-Type": "application/json"})
    assert response.status_code == 413, response.text


def test_streamed_body_over_the_limit_is_refused(monkeypatch):
    # [DOS FIX v2] an upload WITHOUT Content-Length (chunked) must not slip past MAX_BODY_BYTES
    module = load_app(monkeypatch, REQUIRE_AUTH="false", APP_ENV="staging", MAX_BODY_BYTES="64")

    def chunks():
        for _ in range(10):
            yield b"x" * 50

    response = TestClient(module.app).post(ENDPOINT, content=chunks(), headers={"Content-Type": "application/json"})
    assert response.status_code == 413, response.text


def test_key_with_a_non_prediction_role_gets_403(monkeypatch, tmp_path):
    keys = str(tmp_path / "api_keys.json")
    raw_key = create_api_key.create_key(keys, owner="pytest", role="billing")
    module = load_app(monkeypatch, REQUIRE_AUTH="true", API_KEYS_PATH=keys, APP_ENV="production")
    response = TestClient(module.app).post(ENDPOINT, json={}, headers={"X-API-Key": raw_key})
    assert response.status_code == 403, response.text


def test_bad_keys_are_throttled_and_failures_are_uniform(monkeypatch, tmp_path):
    # [BRUTE-FORCE GUARD] guessing keys must hit 429 after AUTH_FAIL_LIMIT_PER_MINUTE failures; an unknown key id, a malformed key and a
    # wrong secret for a REAL key id must all look identical (same status and text) so key ids cannot be enumerated.
    keys = str(tmp_path / "api_keys.json")
    real = create_api_key.create_key(keys, owner="pytest")
    module = load_app(monkeypatch, REQUIRE_AUTH="true", API_KEYS_PATH=keys, APP_ENV="production", AUTH_FAIL_LIMIT_PER_MINUTE="3")
    client = TestClient(module.app)
    wrong_secret = real.split(".", 1)[0] + ".definitely-not-the-secret"
    texts = set()
    for bad in ("nope.nope", "malformed-no-dot", wrong_secret):
        r = client.post(ENDPOINT, json={}, headers={"X-API-Key": bad})
        assert r.status_code == 401, (bad, r.status_code)
        texts.add(r.json()["detail"])
    assert len(texts) == 1, texts
    assert client.post(ENDPOINT, json={}, headers={"X-API-Key": "nope.nope"}).status_code == 429
    # ... but a VALID key is never locked out by someone else's failures
    assert client.post(ENDPOINT, json={}, headers={"X-API-Key": real}).status_code not in (401, 429)


def test_auth_off_caller_is_anonymous_never_admin(monkeypatch):
    module = load_app(monkeypatch, REQUIRE_AUTH="false", APP_ENV="staging")
    assert module.require_api_key(None)["role"] == "anonymous"


def test_health_shows_artifact_names_only(monkeypatch):
    module = load_app(monkeypatch, REQUIRE_AUTH="false", APP_ENV="staging")
    module._ARTIFACT_LOAD_ERRORS["demo"] = "ValueError: /secret/path leaked"
    body = TestClient(module.app).get("/health").json()
    assert body["artifact_load_errors"] == ["demo"], body


def test_rate_limit_uses_forwarded_ip_only_with_trusted_proxy_hops(monkeypatch):
    module = load_app(monkeypatch, REQUIRE_AUTH="false", APP_ENV="staging", TRUSTED_PROXY_HOPS="1", RATE_LIMIT_PER_MINUTE="1")
    client = TestClient(module.app)
    first = [client.post(ENDPOINT, json={}, headers={"X-Forwarded-For": "1.1.1.1"}).status_code for _ in range(2)]
    other = client.post(ENDPOINT, json={}, headers={"X-Forwarded-For": "2.2.2.2"}).status_code
    assert first[0] != 429 and first[1] == 429 and other != 429, (first, other)


def test_missing_key_store_is_a_graceful_startup_error(monkeypatch, tmp_path, capsys):
    # [RULE 2] auth required but no key store anywhere: the import must NOT crash; a StartupError is logged with the exact message,
    # prediction answers 503, /health is 503 (auth_ready=false) and /healthz (liveness) stays 200 so the pod is not restart-looped.
    module = load_app(monkeypatch, REQUIRE_AUTH="true", API_KEYS_PATH=str(tmp_path / "missing.json"), APP_ENV="production")
    assert "Authentication enabled but no valid key-store or env found at runtime." in capsys.readouterr().out
    assert isinstance(module._AUTH_STARTUP_ERROR, module.StartupError)
    client = TestClient(module.app)
    assert client.get("/healthz").status_code == 200
    health = client.get("/health")
    assert health.status_code == 503 and health.json().get("auth_ready") is False, health.text
    assert client.post(ENDPOINT, json={}, headers={"X-API-Key": "a.b"}).status_code == 503


def test_api_keys_json_environment_variable_is_a_key_store(monkeypatch, tmp_path):
    # [RULE 2] no mounted file, but the key store arrives as the API_KEYS_JSON environment variable (e.g. a k8s Secret via secretKeyRef)
    keys = str(tmp_path / "api_keys.json")
    raw_key = create_api_key.create_key(keys, owner="pytest")
    with open(keys, "r", encoding="utf-8") as f:
        payload = f.read()
    module = load_app(monkeypatch, REQUIRE_AUTH="true", API_KEYS_PATH=str(tmp_path / "no-such-file.json"), API_KEYS_JSON=payload,
                      APP_ENV="production")
    assert module._AUTH_STARTUP_ERROR is None
    client = TestClient(module.app)
    assert client.post(ENDPOINT, json={}, headers={"X-API-Key": "nope.nope"}).status_code == 401
    accepted = client.post(ENDPOINT, json={}, headers={"X-API-Key": raw_key})
    assert accepted.status_code not in (401, 403) and "key-store" not in accepted.text, accepted.text


def test_invalid_api_keys_json_fails_closed_without_leaking_it(monkeypatch, tmp_path):
    module = load_app(monkeypatch, REQUIRE_AUTH="true", API_KEYS_PATH=str(tmp_path / "none.json"), API_KEYS_JSON="{ SECRET-NOT-JSON",
                      APP_ENV="production")
    response = TestClient(module.app).post(ENDPOINT, json={}, headers={"X-API-Key": "a.b"})
    assert response.status_code == 503 and "SECRET" not in response.text, response.text


def test_monitoring_runs_in_a_background_task(monkeypatch):
    # [RULE 4] latency / drift bookkeeping is queued as a BackgroundTask by check_rate_limit -- never computed in the request path --
    # and the sliding window size comes from DRIFT_WINDOW_SIZE (default 200).
    import inspect
    import time as _time
    module = load_app(monkeypatch, REQUIRE_AUTH="false", APP_ENV="staging", RATE_LIMIT_PER_MINUTE="0")
    assert "background_tasks" in inspect.signature(module.check_rate_limit).parameters
    assert module._DRIFT_WINDOW_SIZE == 200
    assert "_record_and_check_drift" not in inspect.getsource(module)
    module._record_observation(_time.perf_counter() - 0.01)     # exactly what the BackgroundTask runs after a response
    snapshot = TestClient(module.app).get("/monitoring").json()
    assert snapshot["window_size"] == 200 and snapshot["latency_ms"]["count"] >= 1, snapshot
    assert load_app(monkeypatch, DRIFT_WINDOW_SIZE="50", REQUIRE_AUTH="false", APP_ENV="staging")._DRIFT_WINDOW_SIZE == 50


def test_calendar_year_feature_is_not_drift_monitored(monkeypatch):
    # [CALENDAR-YEAR DRIFT FIX] a *_year feature seen with 2 years in training must not keep X-Model-Drift-Warning on forever.
    module = load_app(monkeypatch, REQUIRE_AUTH="false", APP_ENV="staging", RATE_LIMIT_PER_MINUTE="0")
    if not hasattr(module, "_drift_feature_monitored"):
        return      # this model type ships no histogram drift baseline
    import numpy as np
    def _baseline(first_name):
        return {"format": "histogram_v1", "n_rows": 1000, "feature_names": [first_name, "amt"],
                "features": [{"edges": [0.0], "counts": [500, 500]}, {"edges": [-0.5, 0.0, 0.5], "counts": [250, 250, 250, 250]}]}
    window = np.column_stack([np.full(200, 7.0), np.tile([-1.0, -0.25, 0.25, 1.0], 50)])    # every row is in the newest year
    monkeypatch.setattr(module, "_drift_baseline", _baseline("event_year"))
    monkeypatch.setattr(module, "_drift_prepared", [None])
    assert module._drift_evaluate(window)[0] < 0.2
    monkeypatch.setattr(module, "_drift_baseline", _baseline("event_score"))      # same shift on an ordinary feature IS drift
    monkeypatch.setattr(module, "_drift_prepared", [None])
    assert module._drift_evaluate(window)[0] > 0.2
