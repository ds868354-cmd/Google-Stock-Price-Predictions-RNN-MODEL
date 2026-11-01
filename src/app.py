"""
app.py -- AUTO-GENERATED production serving script for RNN.
This file contains ONLY the code needed to serve this one model type.

Scores the exported ONNX LSTM/GRU with ONNX Runtime. Numeric mode takes a trailing window of floats;
text mode tokenises with the training-time word-to-index vocab.
Endpoint: POST /predict/sequence

Prediction endpoints are ASYNC and run the CPU-bound scoring step via asyncio.to_thread so the event
loop stays responsive under concurrent request load.
"""
import asyncio
import collections
import glob
import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
import threading
import time

import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException, Response, Header, Depends, Request
from pydantic import BaseModel

# [DOCS FIX] /docs, /redoc and /openapi.json are served WITHOUT an API key (they are not behind require_api_key), so in
# production they handed every visitor the full request schema + feature names. They are now off whenever APP_ENV is
# production/prod; set ENABLE_DOCS=true to switch them back on (or run with APP_ENV=staging/dev locally).
_DOCS_ON = (os.environ.get("ENABLE_DOCS", "").strip().lower() in ("1", "true", "yes", "on")
            or os.environ.get("APP_ENV", "").strip().lower() not in ("production", "prod"))
_DOCS_KW = {} if _DOCS_ON else {"docs_url": None, "redoc_url": None, "openapi_url": None}


app = FastAPI(title="Model Serving API (RNN)", **_DOCS_KW)

MODEL_TYPE = "RNN"
# [DEPLOYMENT FIX] Every *_PATH below used to be the FULL path recorded at training time (e.g.
# "/content/output/rnn_model.onnx" from a Colab run) baked in verbatim -- correct on the training
# machine, but Docker/k8s copy artifacts next to THIS file (e.g. /app/), not into the original
# output_dir, so os.path.isfile(...) was always False at serving time and every artifact silently
# failed to load. generate_app_py() now passes only each artifact's basename (see bundler._bundle_relpath),
# and every path below is resolved against this script's own directory instead -- the identical bundle
# then works unmodified from Colab, a bare checkout, or a container, with no environment-specific path
# ever baked in. BUNDLE_DIR_OVERRIDE (env var) is an escape hatch for the rare case artifacts are
# deliberately staged somewhere else.
_BUNDLE_DIR = os.environ.get("BUNDLE_DIR_OVERRIDE") or os.path.dirname(os.path.abspath(__file__))


def _artifact_roots():
    """Directories searched for artifacts: the bundle directory, then its artifacts/ sub-folder. The Dockerfile copies the whole
    (allow-listed) build context, so a flat bundle and one that keeps its weights under artifacts/ both work."""
    return (_BUNDLE_DIR, os.path.join(_BUNDLE_DIR, "artifacts"))


def _resolve_bundle_path(name, default=None):
    """[PATH-RESOLUTION FIX] Resolve an artifact path against this script's own directory (then its artifacts/ sub-folder).

    * `name` given   -> <bundle dir>/<name>, or <bundle dir>/artifacts/<name> when only that one exists.
    * `name` empty   -> the caller did not know the artifact at generation time (e.g. the bundle was rebuilt
      from a folder, or the file was added later). Instead of returning "" (artifact silently never loaded)
      or the bare bundle directory, look for `default` -- an exact filename, a glob, or a tuple of either -- in the
      bundle dir and in artifacts/.
    * Nothing matches -> "" (callers already treat "" as "artifact not shipped").
    The result is therefore either "" or an existing file/directory, never the bundle directory itself."""
    if name:
        for _root in _artifact_roots():
            _p = os.path.join(_root, name)
            if os.path.exists(_p):
                return _p
        return os.path.join(_BUNDLE_DIR, name)
    for pattern in ((default,) if isinstance(default, str) else (default or ())):
        for _root in _artifact_roots():
            candidate = os.path.join(_root, pattern)
            is_glob = any(c in pattern for c in "*?[")
            hits = [h for h in (sorted(glob.glob(candidate)) if is_glob else ([candidate] if os.path.exists(candidate) else []))
                    if os.path.abspath(h) != os.path.abspath(_root)]
            if is_glob and len(hits) > 1:
                # never guess between several candidate files -- serving the wrong model is worse than not starting
                print("[Startup] WARNING: %d files match %s in the bundle (%s) -- ambiguous, none loaded. Ship exactly one, "
                      "or regenerate app.py so it names the file." % (len(hits), pattern,
                                                                      ", ".join(os.path.basename(h) for h in hits)))
                continue
            if hits:
                return hits[0]
    return ""

API_KEYS_PATH = _resolve_bundle_path("api_keys.json", "api_keys.json")
# [SECRETS FIX] api_keys.json is NOT baked into the Docker image / committed to git: a deployer mounts it (k8s Secret volume,
# `docker run -v`) and points API_KEYS_PATH at the mounted file. The copy next to app.py stays the default for local runs.
_API_KEYS_PATH_ENV = os.environ.get("API_KEYS_PATH", "").strip()
if _API_KEYS_PATH_ENV:
    API_KEYS_PATH = _API_KEYS_PATH_ENV


def _env_int(name, default):
    """[ENV-PARSE FIX] a typo such as RATE_LIMIT_PER_MINUTE=sixty used to raise ValueError at import time -> the pod crashed
    in a restart loop before it could even answer /health. A bad value now falls back to the default, with a startup warning."""
    raw = os.environ.get(name, "")
    if not str(raw).strip():
        return default
    try:
        return int(float(str(raw).strip()))
    except ValueError:
        print("[Startup] WARNING: " + name + "=" + repr(raw)[:40] + " is not a number -- using the default " + str(default) + ".")
        return default


# [LOAD-ROBUSTNESS FIX] a truncated / hand-edited / wrong-version JSON artifact used to raise at IMPORT time (KeyError, JSONDecodeError)
# -> uvicorn never bound a port and the pod sat in CrashLoopBackOff with the cause buried in the last log lines. Artifacts are now
# loaded through this helper: a failure is logged, remembered in _ARTIFACT_LOAD_ERRORS, shown on /health (503 + the reason) and
# the app still starts, so `kubectl describe` / the readiness probe point straight at the broken file.
_ARTIFACT_LOAD_ERRORS = {}


def _load_json_artifact(path, label, build=None):
    if not (path and os.path.isfile(path)):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return build(data) if build else data
    except Exception as e:  # noqa: BLE001 -- any parse / schema problem is reported, never raised at import time
        _ARTIFACT_LOAD_ERRORS[label] = type(e).__name__ + ": " + str(e)[:200]
        print("[Startup] WARNING: could not load " + str(path) + " -- " + _ARTIFACT_LOAD_ERRORS[label])
        return None


# [DOS FIX v2] nothing bounded the request body: one multi-GB POST (or a huge base64 image) was read fully into memory. The old guard only
# looked at the Content-Length HEADER, so a chunked / streamed upload (no header, or a lying one) bypassed it. This pure-ASGI middleware
# refuses a declared length above MAX_BODY_BYTES up front AND counts the bytes of a streamed body as they arrive (HTTP 413 either way).
# The default is sized per model type at generation time (small for JSON-only models, larger for image uploads); override with the env var.
_MAX_BODY_DEFAULT = 5000000
MAX_BODY_BYTES = _env_int("MAX_BODY_BYTES", _MAX_BODY_DEFAULT)


async def _send_413(send):
    _body = json.dumps(dict(detail="Request body too large (limit " + str(MAX_BODY_BYTES) + " bytes).")).encode("utf-8")
    await send(dict(type="http.response.start", status=413,
                    headers=[(b"content-type", b"application/json"), (b"content-length", str(len(_body)).encode("ascii"))]))
    await send(dict(type="http.response.body", body=_body))


class _BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        _cl = ""
        for _k, _v in scope.get("headers") or []:
            if _k == b"content-length":
                _cl = _v.decode("latin-1").strip()
                break
        if _cl.isdigit() and int(_cl) > MAX_BODY_BYTES:
            await _send_413(send)
            return
        _seen = [0]

        async def _limited_receive():
            _msg = await receive()
            if _msg["type"] == "http.request":
                _seen[0] += len(_msg.get("body", b""))
                if _seen[0] > MAX_BODY_BYTES:
                    # an HTTPException raised while the route reads the body is turned into a clean 413 JSON response
                    raise HTTPException(413, "Request body too large (limit " + str(MAX_BODY_BYTES) + " bytes).")
            return _msg

        await self.app(scope, _limited_receive, send)


app.add_middleware(_BodyLimitMiddleware)

# [HEALTH REDACTION] /health is unauthenticated (probes need it), so the raw exception text of a failed artifact load (file paths,
# parser messages) must not be handed to every visitor: only the artifact LABELS are shown. The full reason is always in the pod log;
# HEALTH_VERBOSE=true restores the detailed text for a staging debug session.
_HEALTH_VERBOSE = os.environ.get("HEALTH_VERBOSE", "").strip().lower() in ("1", "true", "yes", "on")


def _public_err(msg):
    """Text that may be shown to a caller. Raw exception messages (library internals, file paths) stay in the pod log unless
    HEALTH_VERBOSE=true; used by every unauthenticated / client-facing error that used to echo them."""
    if msg is None:
        return None
    return str(msg)[:300] if _HEALTH_VERBOSE else "details are in the server log (set HEALTH_VERBOSE=true to show them here)"


# ======================================================================================================================
# [RULE 4] NON-BLOCKING MONITORING -- latency, sliding windows and PSI drift run AFTER the response, never inside it.
# check_rate_limit() (every prediction endpoint depends on it) stamps the start time and queues _observe_request() as a FastAPI
# BackgroundTask: Starlette runs it once the response has been sent. The request path therefore pays only for one perf_counter()
# call and one list append -- no window copy, no histogram, no PSI. The windows are fixed-size deques (DRIFT_WINDOW_SIZE, default 200).
# ======================================================================================================================
def _env_float(name, default):
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return float(default)


_DRIFT_WINDOW_SIZE = max(10, _env_int("DRIFT_WINDOW_SIZE", 200))
_DRIFT_MIN_WINDOW = max(10, min(_DRIFT_WINDOW_SIZE, _env_int("DRIFT_MIN_WINDOW", 100)))    # PSI on fewer rows is noise
_DRIFT_EVAL_EVERY = max(1, _env_int("DRIFT_EVAL_EVERY", max(1, _DRIFT_WINDOW_SIZE // 4)))  # PSI recomputed at most once per N rows
_DRIFT_PSI_THRESHOLD = _env_float("DRIFT_PSI_THRESHOLD", 0.2)
_LOG_LATENCY = os.environ.get("LOG_LATENCY", "").strip().lower() in ("1", "true", "yes", "on")

_monitor_lock = threading.Lock()
_latency_window = collections.deque(maxlen=_DRIFT_WINDOW_SIZE)     # seconds, newest last
_drift_rows = collections.deque(maxlen=_DRIFT_WINDOW_SIZE)         # preprocessed feature rows (tabular-style models only)
_monitor_counters = {"observed": 0, "since_eval": 0}
_drift_state = {"drifted": False, "max_psi": None, "worst_feature_index": None, "window_rows": 0, "evaluated_at": None}
_drift_eval_lock = threading.Lock()
_drift_evaluator = [None]    # set by the tabular drift block: callable(2-D array) -> (max_psi, worst_feature_index)


def _evaluate_drift(snapshot):
    """Runs the PSI maths on a snapshot of the window. Background only; one evaluation at a time (a busy evaluator is skipped)."""
    _ev = _drift_evaluator[0]
    if _ev is None or not _drift_eval_lock.acquire(blocking=False):
        return
    try:
        _arr = np.asarray(snapshot, dtype=np.float64)
        _max_psi, _worst = _ev(_arr)
        with _monitor_lock:
            _drift_state.update(drifted=bool(_max_psi > _DRIFT_PSI_THRESHOLD), max_psi=round(float(_max_psi), 4),
                                worst_feature_index=int(_worst), window_rows=int(_arr.shape[0]), evaluated_at=time.time())
    except Exception as _e:  # noqa: BLE001 -- monitoring must never affect serving
        print("[Monitor] drift evaluation failed (ignored): " + type(_e).__name__ + ": " + str(_e)[:200])
    finally:
        _drift_eval_lock.release()


def _record_observation(started, row=None):
    """BACKGROUND work for one served request: latency into the sliding window, the feature row into the drift window, and -- every
    DRIFT_EVAL_EVERY rows once DRIFT_MIN_WINDOW rows exist -- a PSI evaluation."""
    try:
        _latency = max(0.0, time.perf_counter() - started) if started is not None else None
        if row is not None:
            row = np.asarray(row, dtype=np.float64).ravel().tolist()
        _snapshot = None
        with _monitor_lock:
            if _latency is not None:
                _latency_window.append(_latency)
            _monitor_counters["observed"] += 1
            if row is not None and _drift_evaluator[0] is not None:
                _drift_rows.append(row)
                _monitor_counters["since_eval"] += 1
                if len(_drift_rows) >= _DRIFT_MIN_WINDOW and _monitor_counters["since_eval"] >= _DRIFT_EVAL_EVERY:
                    _monitor_counters["since_eval"] = 0
                    _snapshot = list(_drift_rows)
        if _LOG_LATENCY and _latency is not None:
            print("[Latency] %.2f ms" % (_latency * 1000.0))
        if _snapshot is not None:
            _evaluate_drift(_snapshot)
    except Exception as _e:  # noqa: BLE001
        print("[Monitor] observation failed (ignored): " + type(_e).__name__ + ": " + str(_e)[:200])


def _observe_request(request):
    """Background task body: pick up what the request path left on request.state and record it."""
    _st = getattr(request, "state", None)
    _record_observation(getattr(_st, "monitor_t0", None), getattr(_st, "monitor_row", None))


def _monitoring_snapshot():
    with _monitor_lock:
        _lat = sorted(_latency_window)
        _observed = _monitor_counters["observed"]
        _drift = dict(_drift_state)
    def _pct(p):
        return round(_lat[min(len(_lat) - 1, int(p * len(_lat)))] * 1000.0, 3) if _lat else None
    _names = None
    _rec = globals().get("_recipe")
    if isinstance(_rec, dict) and _drift.get("worst_feature_index") is not None:
        _fn = _rec.get("feature_names") or []
        if 0 <= _drift["worst_feature_index"] < len(_fn):
            _names = _fn[_drift["worst_feature_index"]]
    _drift["worst_feature"] = _names
    _drift.update(monitoring_active=_drift_evaluator[0] is not None, psi_threshold=_DRIFT_PSI_THRESHOLD, min_window=_DRIFT_MIN_WINDOW)
    return {"window_size": _DRIFT_WINDOW_SIZE, "observations_total": _observed,
            "latency_ms": {"count": len(_lat), "p50": _pct(0.50), "p95": _pct(0.95), "max": round(_lat[-1] * 1000.0, 3) if _lat else None},
            "drift": _drift}


# ======================================================================================================================
# [RULE 3] SAFE ONNX + LARGE-WEIGHT LOADING -- companion-file (.onnx.data / sharded weights) resolution that never crashes the import.
# ======================================================================================================================
_ORT_PROVIDERS_ENV = [p.strip() for p in os.environ.get("ORT_PROVIDERS", "CPUExecutionProvider").split(",") if p.strip()]


def _onnx_external_names(model_path):
    """File names the ONNX graph asks for as external data (e.g. 'model.onnx.data'). They are plain strings inside the protobuf, so a
    bounded byte scan finds them without the `onnx` package. Only used on the rare repair path."""
    try:
        if os.path.getsize(model_path) > 256 * 1024 * 1024:
            return []
        with open(model_path, "rb") as _f:
            _blob = _f.read()
        _names = {m.decode("ascii", "ignore") for m in re.findall(rb"[A-Za-z0-9_.\-]{1,200}\.(?:onnx\.data|onnx_data|data)(?![A-Za-z0-9_])", _blob)}
        return sorted(n for n in _names if "/" not in n)
    except OSError:
        return []


def _stage_onnx(model_path, data_src, needed_name, mode):
    """Writable staging directory (TMPDIR) holding the model and its external data under the name the graph expects -- for a read-only
    root filesystem (the Kubernetes default here) where nothing can be repaired next to the model. mode 'link' symlinks both files
    (no extra disk); mode 'copy' copies them (when the runtime refuses symlinked external data)."""
    _stage = tempfile.mkdtemp(prefix="onnx_stage_")
    _sm, _sd = os.path.join(_stage, os.path.basename(model_path)), os.path.join(_stage, needed_name)
    try:
        if mode == "link":
            os.symlink(os.path.abspath(model_path), _sm)
            os.symlink(os.path.abspath(data_src), _sd)
        else:
            _need = os.path.getsize(model_path) + os.path.getsize(data_src)
            if shutil.disk_usage(_stage).free < _need * 1.1:
                raise OSError("not enough free space in the staging directory")
            shutil.copyfile(model_path, _sm)
            shutil.copyfile(data_src, _sd)
    except OSError:
        shutil.rmtree(_stage, ignore_errors=True)
        raise
    return _sm


def _load_onnx_session(ort_module, model_path):
    """Returns (session, None) or (None, reason) and NEVER raises, so a missing / renamed / unreadable ONNX file or companion can no
    longer crash the import (CrashLoopBackOff before uvicorn binds a port). Steps: (1) load as shipped; (2) when that fails because the
    external-data companion has a different name than the graph asks for ('rnn_model_onnx.data' vs 'rnn_model.onnx.data'), and exactly one
    other *.data / *.onnx_data file sits next to the model, give it the expected name -- in place (hard link, then copy) when the
    directory is writable, otherwise through a staging directory in TMPDIR (symlinks, then copies). Zero or several candidates are never
    guessed between."""
    if ort_module is None:
        return None, "onnxruntime is not installed in this image"
    if not (model_path and os.path.isfile(model_path)):
        return None, "ONNX model file not found: " + os.path.basename(str(model_path))
    try:
        _avail = ort_module.get_available_providers()
    except Exception:  # noqa: BLE001
        _avail = ["CPUExecutionProvider"]
    _providers = [p for p in _ORT_PROVIDERS_ENV if p in _avail] or ["CPUExecutionProvider"]

    def _try(path):
        try:
            return ort_module.InferenceSession(path, providers=_providers), None
        except Exception as _e:  # noqa: BLE001
            return None, type(_e).__name__ + ": " + str(_e)[:300]

    _sess, _err = _try(model_path)
    if _sess is not None:
        return _sess, None
    _errors = [_err]
    _dir = os.path.dirname(model_path)
    _wanted = _onnx_external_names(model_path) or [os.path.basename(model_path) + ".data"]
    if len(_wanted) != 1:
        return None, _errors[0] + " | external-data repair skipped: the graph references %d data files" % len(_wanted)
    _needed = _wanted[0]
    _needed_path = os.path.join(_dir, _needed)
    if os.path.isfile(_needed_path):
        return None, _errors[0]            # the companion is there under the right name: the failure is something else
    _cands = sorted(set(glob.glob(os.path.join(_dir, "*.data")) + glob.glob(os.path.join(_dir, "*.onnx_data"))))
    if len(_cands) != 1:
        return None, _errors[0] + " | external-data repair skipped: %d candidate file(s) next to the model" % len(_cands)
    _src = _cands[0]
    for _how in ("hardlink", "copy"):                       # 1) repair in place (writable filesystem)
        try:
            if _how == "hardlink":
                os.link(_src, _needed_path)
            else:
                shutil.copyfile(_src, _needed_path)
        except OSError:
            continue
        _sess, _err = _try(model_path)
        if _sess is not None:
            print("[Startup] Self-heal: ONNX external-data file '%s' provided as '%s' (%s)" % (os.path.basename(_src), _needed, _how))
            return _sess, None
        _errors.append(_err)
        try:
            os.remove(_needed_path)
        except OSError:
            pass
    for _mode in ("link", "copy"):                           # 2) read-only filesystem -> stage in TMPDIR
        try:
            _staged = _stage_onnx(model_path, _src, _needed, _mode)
        except OSError as _e:
            _errors.append("staging (%s) failed: %s" % (_mode, str(_e)[:120]))
            continue
        _sess, _err = _try(_staged)
        if _sess is not None:
            print("[Startup] Self-heal: ONNX model staged in %s with '%s' as '%s' (%s)" % (os.path.dirname(_staged), os.path.basename(_src), _needed, _mode))
            return _sess, None
        _errors.append(_err)
        shutil.rmtree(os.path.dirname(_staged), ignore_errors=True)
    return None, " | ".join(e for e in _errors if e)[:700]


def _check_weight_dir(path):
    """Problems (list of strings; empty = fine) with a local transformers / peft / sentence-transformers weights directory: no weight
    file at all, or a sharded checkpoint whose index lists shard files that are missing. No weight file is opened here."""
    if not os.path.isdir(path):
        return ["directory not found: " + os.path.basename(str(path))]
    _names = os.listdir(path)
    _probs = []
    if not [n for n in _names if n.endswith((".safetensors", ".bin", ".pt", ".pth"))]:
        _probs.append("no weight file (*.safetensors / *.bin) in " + os.path.basename(str(path)))
    for _n in _names:
        if _n.endswith(".index.json"):
            try:
                with open(os.path.join(path, _n), "r", encoding="utf-8") as _f:
                    _wm = json.load(_f).get("weight_map", {})
            except (OSError, ValueError, AttributeError):
                _probs.append("unreadable shard index " + _n)
                continue
            _missing = sorted({v for v in _wm.values() if isinstance(v, str) and not os.path.isfile(os.path.join(path, v))})
            if _missing:
                _probs.append("shard file(s) listed in %s are missing: %s" % (_n, ", ".join(_missing[:5])))
    return _probs


def _hf_safe_kwargs(path):
    """use_safetensors=True whenever the local directory ships *.safetensors, so a pickle-based *.bin next to it is never chosen."""
    try:
        if os.path.isdir(path) and any(n.endswith(".safetensors") for n in os.listdir(path)):
            return {"use_safetensors": True}
    except OSError:
        pass
    return {}



# ---- API-key authentication -------------------------------------------
# SECURITY FIX: every prediction endpoint used to have NO
# authentication at all -- auth.py already implemented APIKeyStore
# (salted HMAC-SHA256 key hashing, never plaintext) and even shipped a
# FASTAPI_AUTH_DEPENDENCY_SNIPPET meant to be wired into this file, but
# generate_app_py() never actually called it, so every generated app.py
# served predictions to anyone who could reach the port. This closes
# that gap without pulling in auth.py's file-locking machinery (that
# module's write-path concurrency guarantees matter for issuing/revoking
# keys, not for this read-only verification at request time): this file
# reads api_keys.json directly and verifies the same salted-hash format
# APIKeyStore.create_key() writes.
#
# Provisioning a key (run once, from the bundle folder -- create_api_key.py ships with the bundle):
#   python create_api_key.py --owner my-service
#   # -> save the printed key now; it is never recoverable from api_keys.json again
#
# Callers then send it as:  -H "X-API-Key: <raw_key>"
#
# REQUIRE_AUTH is tri-state (true / false / auto) -- see the block below.
# [AUTH FALLBACK FIX] REQUIRE_AUTH is now tri-state instead of "true unless exactly 'false'":
#   REQUIRE_AUTH=true  (or 1/yes/on)  -> always enforce. Missing api_keys.json => clear 503 (server misconfigured),
#                                        not a confusing 401 for every caller.
#   REQUIRE_AUTH=false (or 0/no/off)  -> never enforce (local / staging).
#   unset or "auto"                   -> enforce when api_keys.json exists and holds at least one key; if it does
#                                        not exist, fall back to open access with a loud startup warning -- EXCEPT
#                                        when APP_ENV=production, which stays fail-closed.
# /health always reports the effective value as "auth_enforced" so the mode is never a surprise.
_REQUIRE_AUTH_RAW = os.environ.get("REQUIRE_AUTH", "auto").strip().lower() or "auto"
_APP_ENV = os.environ.get("APP_ENV", "").strip().lower()


# [RULE 2] SECRET SOURCES + GRACEFUL FAILURE. The key store is read from a MOUNTED FILE (k8s Secret volume / `docker run -v`, pointed at by
# API_KEYS_PATH) or, when no file exists, from the API_KEYS_JSON ENVIRONMENT VARIABLE (the same JSON as api_keys.json, e.g. a k8s Secret
# exposed with secretKeyRef). Nothing is ever baked into the image. If authentication is required and neither source yields a valid
# key store, the app does NOT crash: it logs a StartupError, keeps running, answers 503 on the prediction endpoints and reports
# auth_ready=false (HTTP 503) on /health until a key store is provided.
class StartupError(RuntimeError):
    """A mandatory runtime secret is missing. Created and LOGGED at start-up (see _startup_auth_check), never raised at import."""


_AUTH_STORE_MSG = "Authentication enabled but no valid key-store or env found at runtime."
_API_KEYS_ENV_RAW = os.environ.get("API_KEYS_JSON", "").strip()
_env_keys_cache = [None]


def _keys_from_env():
    """Key dict parsed from API_KEYS_JSON (empty when unset). Accepts the api_keys.json layout (an object with a "keys" object) or a
    bare object mapping key ids to entries. The error text never contains the variable's content."""
    if not _API_KEYS_ENV_RAW:
        return {}
    if _env_keys_cache[0] is None:
        try:
            _data = json.loads(_API_KEYS_ENV_RAW)
        except ValueError:
            raise ValueError("API_KEYS_JSON is not valid JSON")
        _keys = _data.get("keys", _data) if isinstance(_data, dict) else None
        if not isinstance(_keys, dict):
            raise ValueError("API_KEYS_JSON has no valid 'keys' object")
        _env_keys_cache[0] = _keys
    return _env_keys_cache[0]


def _api_keys_file_has_keys():
    if not (API_KEYS_PATH and os.path.isfile(API_KEYS_PATH)):
        if _API_KEYS_ENV_RAW:
            try:
                return bool(_keys_from_env())
            except ValueError:
                print("[Startup] WARNING: API_KEYS_JSON is set but is not valid JSON -- authentication stays ENFORCED and every "
                      "request gets 503 until it is fixed.")
                return True
        return False
    try:
        with open(API_KEYS_PATH, "r", encoding="utf-8") as f:
            return bool(json.load(f).get("keys"))
    except (OSError, ValueError, AttributeError):
        # [CORRUPT api_keys.json FIX] a file that EXISTS but cannot be parsed must never silently switch authentication
        # off: treat it as "keys configured" so auth stays enforced (require_api_key then answers a clear 503).
        print("[Startup] WARNING: " + str(API_KEYS_PATH) + " exists but is unreadable / not valid JSON -- authentication "
              "stays ENFORCED and every request gets 503 until the file is fixed or re-provisioned.")
        return True


def _resolve_require_auth():
    if _REQUIRE_AUTH_RAW in ("false", "0", "no", "off"):
        return False
    if _REQUIRE_AUTH_RAW in ("true", "1", "yes", "on"):
        return True
    if _api_keys_file_has_keys():
        return True
    if _APP_ENV in ("production", "prod"):
        return True
    print("[Startup] WARNING: no api_keys.json (or no keys in it) found at "
          + (API_KEYS_PATH or "<bundle dir>/api_keys.json")
          + " -- running with authentication DISABLED. Set REQUIRE_AUTH=true and provision a key before exposing "
            "this service beyond local/staging use, or set APP_ENV=production to refuse to start open.")
    return False


REQUIRE_AUTH = _resolve_require_auth()
if not REQUIRE_AUTH and _APP_ENV in ("production", "prod") and _REQUIRE_AUTH_RAW in ("false", "0", "no", "off"):
    print("[Startup] WARNING: APP_ENV=production with REQUIRE_AUTH=false -- the prediction endpoints are OPEN to every caller. "
          "Only do this behind a gateway that authenticates requests.")


# [KEY-FILE CACHE] api_keys.json used to be opened + parsed on EVERY request -- including every unauthenticated guess, so a flood of bad
# keys was a cheap disk/CPU DoS. The parsed file is now cached and re-read only when its (mtime, size) changes, so a rotated / revoked key
# (or a k8s Secret update, which swaps the file) still takes effect on the next request.
_api_keys_cache = [None, None]   # [(mtime_ns, size), parsed keys dict]
_api_keys_cache_lock = threading.Lock()


def _load_api_keys():
    if not (API_KEYS_PATH and os.path.isfile(API_KEYS_PATH)):
        return _keys_from_env()          # no mounted file: the API_KEYS_JSON environment variable (or {} when unset)
    try:
        _st = os.stat(API_KEYS_PATH)
        _sig = (_st.st_mtime_ns, _st.st_size)
        with _api_keys_cache_lock:
            if _api_keys_cache[0] == _sig and _api_keys_cache[1] is not None:
                return _api_keys_cache[1]
        with open(API_KEYS_PATH, "r", encoding="utf-8") as f:
            keys = json.load(f).get("keys", {})
    except (OSError, ValueError, AttributeError) as e:
        raise ValueError("api_keys.json is unreadable or not valid JSON: %s" % e)
    if not isinstance(keys, dict):
        raise ValueError("api_keys.json has no valid 'keys' object")
    with _api_keys_cache_lock:
        _api_keys_cache[0], _api_keys_cache[1] = _sig, keys
    return keys


_PREDICT_ROLES = frozenset(r.strip().lower() for r in os.environ.get("PREDICT_ROLES", "reader,predictor,admin").split(",") if r.strip())

# [BRUTE-FORCE / DoS GUARD] failed authentications used to be rejected BEFORE the rate limiter ever ran, so nothing throttled guessing.
# Each client IP may now fail AUTH_FAIL_LIMIT_PER_MINUTE (default 20) times a minute; further failures get 429 (and Retry-After) instead of
# 401 until the window clears (0 disables). Only 401s count -- a valid key with the wrong role (403) or a server-side problem (503) is not
# the caller guessing. A VALID key is never blocked by this guard: behind a proxy several clients may share one IP, and an attacker must
# not be able to lock the legitimate ones out by failing on purpose.
_AUTH_FAIL_LIMIT = _env_int("AUTH_FAIL_LIMIT_PER_MINUTE", 20)
_auth_fail_lock = threading.Lock()
_auth_fail_windows = {}   # client ip -> collections.deque of failure timestamps within the last 60 s
_DUMMY_SALT = "0" * 32


def _auth_fail_record(ip):
    """Records one failed attempt; returns True when this IP is now over its failure budget."""
    if _AUTH_FAIL_LIMIT <= 0:
        return False
    now = time.time()
    with _auth_fail_lock:
        if len(_auth_fail_windows) > 50000:
            for _k in [k for k, b in _auth_fail_windows.items() if not b or b[-1] < now - 60.0]:
                del _auth_fail_windows[_k]
            if len(_auth_fail_windows) > 50000:
                _auth_fail_windows.clear()      # attacker-controlled keys: reset rather than grow without bound
        bucket = _auth_fail_windows.setdefault(ip, collections.deque())
        while bucket and bucket[0] < now - 60.0:
            bucket.popleft()
        bucket.append(now)
        return len(bucket) > _AUTH_FAIL_LIMIT


def _verify_api_key(x_api_key):
    if not x_api_key:
        raise HTTPException(401, "Missing X-API-Key header.")
    key_id, _sep, raw_key = str(x_api_key).partition(".")
    try:
        entry = _load_api_keys().get(key_id) if _sep else None
    except ValueError as e:
        # the reason (parser message) goes to the pod log, never to an unauthenticated caller
        print("[Auth] api_keys.json cannot be read: " + str(e))
        raise HTTPException(503, "Authentication is enabled but the key store cannot be read. Check the server log.")
    # [UNIFORM FAILURE] unknown id, malformed key, revoked key and wrong secret all give the SAME 401 text and do the SAME work (one
    # HMAC, against a dummy salt when the id does not exist), so neither the message nor the response time tells a caller which key ids exist.
    _known = isinstance(entry, dict)
    _salt = str(entry.get("salt")) if _known and entry.get("salt") is not None else _DUMMY_SALT
    expected_hash = hmac.new(_salt.encode("utf-8"), (raw_key if _sep else str(x_api_key)).encode("utf-8"), hashlib.sha256).hexdigest()
    try:
        hash_ok = hmac.compare_digest(expected_hash, str(entry.get("hash")) if _known else "0" * 64)
    except (TypeError, ValueError):
        hash_ok = False
    if not (_known and hash_ok and not entry.get("revoked")):
        raise HTTPException(401, "Invalid API key.")
    try:
        _expired = entry.get("expires_at") is not None and time.time() > float(entry["expires_at"])
    except (TypeError, ValueError):
        _expired = True
    if _expired:
        raise HTTPException(401, "API key expired.")
    # [ROLE ENFORCEMENT] the role stored in api_keys.json used to be recorded for audit only: ANY valid key could call the endpoints.
    # Only roles listed in PREDICT_ROLES (default reader,predictor,admin) may score; a key with any other role (e.g. "disabled",
    # "billing") gets 403. With auth OFF the caller is "anonymous" (never "admin").
    _role = str(entry.get("role") or "").strip().lower()
    if _role not in _PREDICT_ROLES:
        raise HTTPException(403, "API key role '" + _role[:30] + "' is not allowed to call the prediction endpoints.")
    return {"key_id": key_id, "owner": entry.get("owner"), "role": _role}


def _auth_store_ok():
    """True when at least one key can be read from the mounted file or from API_KEYS_JSON."""
    try:
        return bool(_load_api_keys())
    except ValueError:
        return False


def _auth_ready():
    """Authentication is either not required, or its key store is present and valid (re-checked on every call: cheap, cached by mtime)."""
    return (not REQUIRE_AUTH) or _auth_store_ok()


def _startup_auth_check():
    """[RULE 2] Logs a StartupError -- and returns it -- when auth is required but no valid key store exists. Never raises."""
    if not REQUIRE_AUTH or _auth_store_ok():
        return None
    _err = StartupError(_AUTH_STORE_MSG)
    print("[StartupError] " + str(_err))
    print("[StartupError] Looked for api_keys.json at " + (API_KEYS_PATH or "<bundle dir>/api_keys.json")
          + (" (API_KEYS_PATH=" + _API_KEYS_PATH_ENV + ")" if _API_KEYS_PATH_ENV else "") + " and for the API_KEYS_JSON environment variable. "
          "The service keeps running: prediction endpoints answer 503 and /health reports auth_ready=false until a key store is mounted "
          "(Kubernetes Secret volume / docker -v) or API_KEYS_JSON is set.")
    return _err


def require_api_key(x_api_key: str = Header(default=None), request: Request = None):
    """FastAPI dependency -- add `identity: dict = Depends(require_api_key)`
    to any endpoint that must be authenticated."""
    if not REQUIRE_AUTH:
        return {"key_id": None, "owner": "anonymous", "role": "anonymous"}
    if not ((API_KEYS_PATH and os.path.isfile(API_KEYS_PATH)) or _API_KEYS_ENV_RAW):
        raise HTTPException(503, _AUTH_STORE_MSG)
    try:
        return _verify_api_key(x_api_key)
    except HTTPException as _e:
        if _e.status_code == 401 and _auth_fail_record(_client_ip(request) if request is not None else "unknown"):
            raise HTTPException(429, "Too many failed authentication attempts. Retry later.", headers={"Retry-After": "60"})
        raise


_AUTH_STARTUP_ERROR = _startup_auth_check()

# ---- Rate limiting (per API key / per IP) -------------------------------
# GAP FIX: nothing in the served API throttled request volume -- one
# noisy or abusive client could exhaust the process. This adds an
# in-memory sliding-window limiter keyed by the authenticated key_id
# (falls back to caller IP when REQUIRE_AUTH=false), so a single caller
# can't starve everyone else sharing this process.
# LIMITATION, stated plainly rather than silently ignored (the shipped k8s-ingress.yaml adds a cluster-wide limit at the ingress to cover it):
# the window is per-process. Behind multiple uvicorn/gunicorn workers or replicas, each
# process throttles independently, so the effective ceiling is
# RATE_LIMIT_PER_MINUTE * worker_count, not RATE_LIMIT_PER_MINUTE. For a
# multi-process deployment, swap _rate_limit_windows for a shared store
# (e.g. Redis INCR + EXPIRE) behind the same check_rate_limit() interface.
RATE_LIMIT_PER_MINUTE = _env_int("RATE_LIMIT_PER_MINUTE", 60)
_rate_limit_lock = threading.Lock()
_rate_limit_windows = {}  # caller key -> collections.deque of timestamps within the last 60s (popleft is O(1); list.pop(0) was O(n))
_rate_limit_last_purge = [0.0]
_RATE_LIMIT_MAX_BUCKETS = 50000   # hard cap on distinct callers kept in memory (purged early when exceeded)
# [PROXY-AWARE IP] behind an ingress / load balancer request.client.host is the PROXY's address, so with auth off every caller shared one
# bucket. TRUSTED_PROXY_HOPS=N (default 0 = ignore X-Forwarded-For, spoof-proof) takes the N-th address from the RIGHT of X-Forwarded-For,
# i.e. the one written by your own outermost trusted proxy. Set it to 1 behind a single ingress controller.
_TRUSTED_PROXY_HOPS = _env_int("TRUSTED_PROXY_HOPS", 0)


def _client_ip(request: Request) -> str:
    if _TRUSTED_PROXY_HOPS > 0:
        _xff = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(_xff) >= _TRUSTED_PROXY_HOPS:
            return _xff[-_TRUSTED_PROXY_HOPS][:64]
    return request.client.host if request.client else "unknown"


def _rate_limit_caller_key(identity: dict, request: Request) -> str:
    if identity.get("key_id"):
        return f"key:{identity['key_id']}"
    return f"ip:{_client_ip(request)}"


def _rate_limit_check(request, identity):
    """The rate limiter proper (see check_rate_limit). Set RATE_LIMIT_PER_MINUTE=0 to disable throttling entirely."""
    if RATE_LIMIT_PER_MINUTE <= 0:
        return identity
    caller_key = _rate_limit_caller_key(identity, request)
    now = time.time()
    window_start = now - 60.0
    with _rate_limit_lock:
        # [MEMORY FIX] with auth off every client IP gets its own bucket and nothing ever removed them -> unbounded growth.
        # Once a minute, drop buckets whose newest request is older than the window.
        if now - _rate_limit_last_purge[0] > 60.0 or len(_rate_limit_windows) > _RATE_LIMIT_MAX_BUCKETS:
            _rate_limit_last_purge[0] = now
            for _k in [k for k, b in _rate_limit_windows.items() if not b or b[-1] < window_start]:
                del _rate_limit_windows[_k]
        bucket = _rate_limit_windows.get(caller_key)
        if bucket is None:
            bucket = _rate_limit_windows[caller_key] = collections.deque()
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_PER_MINUTE:
            retry_after = max(1, int(bucket[0] + 60.0 - now))
            raise HTTPException(
                429,
                f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE} requests/min). Retry later.",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)
    return identity


def check_rate_limit(request: Request, identity: dict = Depends(require_api_key), background_tasks: BackgroundTasks = None):
    """FastAPI dependency -- use in place of `Depends(require_api_key)` on any endpoint that should be both authenticated AND
    rate-limited (it depends on require_api_key itself, so auth still runs first and a 401 still takes priority over a 429).
    [RULE 4] It also stamps the request start time and queues the monitor (latency window + drift) as a BackgroundTask, which runs
    AFTER the response has been sent -- the request path pays for one perf_counter() call."""
    _rate_limit_check(request, identity)
    _st = getattr(request, "state", None)
    if _st is not None:
        _st.monitor_t0 = time.perf_counter()
    if background_tasks is not None:
        background_tasks.add_task(_observe_request, request)
    return identity


try:
    import onnxruntime as ort
except ImportError:
    ort = None

ONNX_MODEL_PATH = _resolve_bundle_path("rnn_model.onnx", "*.onnx")

# [RULE 3] the session is created by _load_onnx_session() (core block): safe try/except, external-data (.onnx.data) name repair that also
# works on a read-only root filesystem, explicit providers. It never raises -- a bad model file leaves _session None and /health reports it.
_session = None
_session_load_error = None
if ONNX_MODEL_PATH and os.path.isfile(ONNX_MODEL_PATH):
    _session, _session_load_error = _load_onnx_session(ort, ONNX_MODEL_PATH)
    if _session_load_error:
        print("[Startup] WARNING: failed to load ONNX model " + os.path.basename(ONNX_MODEL_PATH) + ": " + _session_load_error)


# ---- Task-type resolution (regression | binary | multiclass) -----------------------------------------------------
# [TASK-TYPE FIX] The serving code used to decide "classification vs regression" from a default string
# ("classification") when the artifact did not say -- so a regression LightGBM/XGBoost ensemble was pushed through
# a sigmoid and returned 0/1. The task is now read from the model artifact first, then feature_recipe.json, and the
# prediction shape follows it: regression -> a float; binary/multiclass -> class label + probabilities.
_REGRESSION_TASKS = ("regression", "regressor", "reg", "timeseries", "time_series", "forecast", "forecasting")


def _normalize_task(raw):
    t = str(raw or "").strip().lower().replace("-", "_")
    if t in _REGRESSION_TASKS:
        return "regression"
    if t in ("multiclass", "multi_class", "multiclass_classification"):
        return "multiclass"
    if t in ("binary", "binary_classification"):
        return "binary"
    if t in ("classification", "classifier", "clf"):
        return "classification"
    return ""


_MODEL_TASK_OVERRIDE = _normalize_task(os.environ.get("MODEL_TASK_OVERRIDE", ""))
if os.environ.get("MODEL_TASK_OVERRIDE", "").strip() and not _MODEL_TASK_OVERRIDE:
    print("[Startup] WARNING: MODEL_TASK_OVERRIDE='" + os.environ.get("MODEL_TASK_OVERRIDE", "") + "' is not one of "
          "regression | binary | multiclass | classification -- ignored.")

_model_task_file = None
_MODEL_TASK_FILE = os.path.join(_BUNDLE_DIR, "model_task.json")
if os.path.isfile(_MODEL_TASK_FILE):
    try:
        with open(_MODEL_TASK_FILE, "r", encoding="utf-8") as _f_task:
            _model_task_file = json.load(_f_task)
    except (OSError, ValueError) as _task_err:
        print("[Startup] WARNING: model_task.json exists but is unreadable (" + str(_task_err) + ") -- ignored.")


def _model_task(payload=None):
    """Task of the loaded model, resolved in this order: MODEL_TASK_OVERRIDE env var -> the model artifact's own
    "task" key -> model_task.json (bundle dir) -> feature_recipe.json's "task". "" if none of them says."""
    if _MODEL_TASK_OVERRIDE:
        return _MODEL_TASK_OVERRIDE
    for src in (payload, _model_task_file, globals().get("_recipe")):  # some bundles have no _recipe
        if isinstance(src, dict):
            t = _normalize_task(src.get("task"))
            if t:
                return t
    return ""


def _recipe_class_names():
    rec = globals().get("_recipe")
    names = rec.get("class_names") if isinstance(rec, dict) else None
    return list(names) if names else None


def _softmax(z):
    z = np.asarray(z, dtype=np.float64).ravel()
    e = np.exp(z - z.max())
    return e / e.sum()


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -500.0, 500.0)))


_THR_WARNED = []   # warn about an unusable recipe threshold ONCE, not on every request


def _decision_threshold_cfg():
    """The manual decision threshold this model was BUILT with, from feature_recipe.json: `decision_threshold`
    (TABULAR_ML, tabular_ml_settings.decision_threshold) or `ann_decision_threshold` (ANN, ann_imbalance_settings).
    A dict {threshold, positive_index} (a bare float is accepted too). For a bundle built WITHOUT one the deployer can set
    the DECISION_THRESHOLD env var by hand. None -> no threshold known: binary tabular/ANN scoring then refuses to guess a
    0.50 cut (see _threshold_missing_error). Out-of-range values are ignored with a warning."""
    _rec = globals().get("_recipe")
    _dt = None
    if isinstance(_rec, dict):
        _dt = _rec.get("decision_threshold")
        if _dt is None:
            _dt = _rec.get("ann_decision_threshold")
    if str(os.environ.get("DECISION_THRESHOLD", "")).strip():
        # [THRESHOLD OVERRIDE] the deployer's DECISION_THRESHOLD env var wins over the value baked into the recipe, so the
        # precision/recall trade-off can be tuned per deployment (kubectl set env) without retraining. Keeps the recipe's
        # positive_index. Unset the variable to go back to the trained threshold.
        _env_thr = str(os.environ["DECISION_THRESHOLD"]).strip()
        _pos_idx = int(_dt.get("positive_index", 1)) if isinstance(_dt, dict) else 1
        _dt = {"threshold": _env_thr, "positive_index": _pos_idx}
    if isinstance(_dt, dict) and isinstance(_dt.get("threshold"), str):
        try:
            _dt = {**_dt, "threshold": float(_dt["threshold"])}
        except ValueError:
            return None
    if isinstance(_dt, str):
        try:
            _dt = float(_dt)
        except ValueError:
            return None
    if isinstance(_dt, (int, float)) and not isinstance(_dt, bool):
        _dt = {"threshold": float(_dt), "positive_index": 1}
    if not isinstance(_dt, dict) or _dt.get("threshold") is None:
        return None
    try:
        _thr = float(_dt["threshold"])
        _pos = int(_dt.get("positive_index", 1))
    except (TypeError, ValueError):
        return None
    if not (0.0 < _thr < 1.0) or _pos not in (0, 1):
        if not _THR_WARNED:
            _THR_WARNED.append(1)
            print("[Startup] WARNING: decision_threshold in feature_recipe.json is not a float in (0,1) -- ignored.")
        return None
    return {"threshold": _thr, "positive_index": _pos}


def _threshold_class_index(probs):
    """Predicted class INDEX for a binary probability vector under the manual threshold; None when no threshold is set
    or the vector is not binary (then callers keep their normal argmax)."""
    _dt = _decision_threshold_cfg()
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if _dt is None or probs.size != 2:
        return None
    _pos = _dt["positive_index"]
    return _pos if float(probs[_pos]) >= _dt["threshold"] else 1 - _pos


def _is_binary_classifier():
    """True when the loaded model scores exactly two classes -- decided from the recipe, else the model artifact itself (the
    recipe's class_names can be null), so the check works for every backend."""
    rec = globals().get("_recipe")
    if isinstance(rec, dict) and str(rec.get("task", "")).lower() in ("regression", "regressor"):
        return False
    names = _recipe_class_names()
    if names:
        return len(names) == 2
    for _m in (globals().get("_tree_ensemble"), globals().get("_linear_model")):
        if isinstance(_m, dict):
            if _m.get("classes_"):
                return len(_m["classes_"]) == 2
            if str(_m.get("task", "")).lower() == "binary":
                return True
    sess = globals().get("_session")
    if sess is not None:
        try:
            return any(len(o.shape or []) == 2 and o.shape[1] == 2 for o in sess.get_outputs())
        except Exception:  # noqa: BLE001
            return False
    return False


def _threshold_required():
    """Tabular / ANN binary classifiers have NO default cut-off (the automatic 0.50 was removed on purpose): one must come from
    feature_recipe.json or the DECISION_THRESHOLD env var. CNN / RNN apps share this fragment but have no threshold concept."""
    return MODEL_TYPE in ("TABULAR_ML", "ANN") and _is_binary_classifier()


def _threshold_ready():
    return (not _threshold_required()) or _decision_threshold_cfg() is not None


def _threshold_missing_error():
    return HTTPException(500, "This binary classifier has no decision threshold: feature_recipe.json carries none and the "
                              "DECISION_THRESHOLD env var is not set. The automatic 0.50 default was removed -- re-train with a "
                              "manual decision_threshold, or set DECISION_THRESHOLD (a number strictly between 0 and 1, e.g. "
                              "0.35) for this deployment.")


def _classification_result(probs, classes=None, require_threshold=False):
    """probs: 1-D class probabilities. `prediction` is the model's own class label when known (classes_), else the
    class index; `class_name` adds the human-readable name from the recipe when it differs."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    idx = int(np.argmax(probs))
    _dt = _decision_threshold_cfg()
    _t_idx = _threshold_class_index(probs)
    if _t_idx is None and require_threshold and probs.size == 2:   # tabular/ANN binary: no silent 0.50 / argmax cut
        raise _threshold_missing_error()
    if _t_idx is not None:  # binary model built with a manual decision threshold -> the SAME cut-off at serving
        idx = _t_idx
    label = classes[idx] if classes and idx < len(classes) else idx
    if isinstance(label, np.generic):
        label = label.item()
    if isinstance(label, float) and label.is_integer():   # classes_ stored as 0.0 / 1.0 -> answer 0 / 1
        label = int(label)
    result = {"prediction": label, "probability": float(probs[idx]), "probabilities": probs.tolist()}
    names = _recipe_class_names()
    if names and idx < len(names) and names[idx] != label:
        result["class_name"] = names[idx]
    if _dt is not None and probs.size == 2:
        result["decision_threshold"] = float(_dt["threshold"])
    return result


def _with_class_probs(result, raw_output):
    """Image / sequence-text classifiers used to return only the raw logits under `prediction`. That field is left exactly as
    it was (no contract break) and the decoded answer is ADDED: predicted_class + probabilities."""
    cls = _postprocess_onnx_tabular([np.asarray(raw_output)], "classification", require_threshold=False)
    result["predicted_class"] = cls["prediction"]
    result["probabilities"] = cls["probabilities"]
    return result


def _postprocess_onnx_tabular(outputs, task, require_threshold=True):
    """Shape an ONNX Runtime result by task. Regression keeps the old behaviour (raw output as `prediction`).
    Classification: skl2onnx classifiers return [labels, probabilities]; torch MLPs return one logits (or
    sigmoid/softmax) tensor -- both become {prediction, probability, probabilities}."""
    if task in ("regression", ""):  # "" = artifact + recipe silent -> legacy pass-through, never guess classification
        return {"prediction": np.asarray(outputs[0]).tolist()}
    if len(outputs) > 1:
        label = np.asarray(outputs[0]).ravel()[0]
        label = label.item() if isinstance(label, np.generic) else label
        raw = outputs[1]
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            keys = list(raw[0].keys())
            probs = [float(raw[0][k]) for k in keys]
            res = {"prediction": label, "probability": float(max(probs)), "probabilities": probs}
            res["classes"] = [k.item() if isinstance(k, np.generic) else k for k in keys]
            _t_idx = _threshold_class_index(probs)
            if _t_idx is None and require_threshold and len(probs) == 2:
                raise _threshold_missing_error()
            if _t_idx is not None:   # manual threshold: the ONNX label is the default-0.50 label -> re-decide from probs
                _lab = keys[_t_idx]
                res["prediction"] = _lab.item() if isinstance(_lab, np.generic) else _lab
                res["probability"] = float(probs[_t_idx])
                res["decision_threshold"] = _decision_threshold_cfg()["threshold"]
            return res
        probs = np.asarray(raw, dtype=np.float64).reshape(-1)
        res = _classification_result(probs, require_threshold=require_threshold)
        if _threshold_class_index(probs) is None:
            res["prediction"] = label       # no manual threshold: ONNX's own label (default cut) as before
        # with a manual threshold _classification_result() already decided the label from the probabilities; targets are
        # label-encoded 0..k-1, so the class index IS the label the ONNX graph would have emitted.
        return res
    vec = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
    if vec.size == 1:
        p = float(vec[0])
        if not 0.0 <= p <= 1.0:
            p = float(_sigmoid(p))
        return _classification_result([1.0 - p, p], require_threshold=require_threshold)
    if not (np.all(vec >= 0.0) and np.all(vec <= 1.0) and abs(float(vec.sum()) - 1.0) < 1e-3):
        vec = _softmax(vec)
    return _classification_result(vec, require_threshold=require_threshold)


RNN_MODE = "numeric"
RNN_WINDOW_SIZE = 30
RNN_VOCAB_PATH = _resolve_bundle_path("", "rnn_vocab.json")
# [RNN REGRESSION TARGET-SCALING SERVING FIX] see the matching comment in main()'s RNN branch --
# numeric-mode (forecasting) targets are z-scored before training; without this file+path a
# deployed forecast endpoint silently returned z-scored numbers instead of real units.
RNN_TARGET_SCALER_PATH = _resolve_bundle_path("rnn_target_scaler.json", "rnn_target_scaler.json")

# ---- RNN: word-to-index vocab for text mode (see model_zoo.tokenize_text_sequence) ----
_rnn_vocab_data = _load_json_artifact(RNN_VOCAB_PATH, "rnn_vocab.json") or {}
_rnn_vocab = _rnn_vocab_data.get("vocab")
_rnn_max_len = _rnn_vocab_data.get("max_len")

_rnn_target_scaler = _load_json_artifact(RNN_TARGET_SCALER_PATH, "rnn_target_scaler.json")
_MAX_TEXT_CHARS = _env_int("MAX_TEXT_CHARS", 100000)


class SequenceRequest(BaseModel):
    # numeric mode: values = trailing window of floats, length RNN_WINDOW_SIZE
    # text mode: text = a raw string to classify
    values: list = None
    text: str = None


def _score_sequence(values, text):
    """Scores one RNN request: numeric mode takes a raw trailing window of floats
    (no scaling recipe needed -- see build_sequence_tensors's docstring,
    numeric mode never scaled at training time either, so there is
    nothing to mismatch), text mode tokenizes with the SAME whitespace/
    word-regex vocab training used (model_zoo.tokenize_text_sequence),
    padded/truncated to the exact fixed length the ONNX graph was
    traced with (RNN_MAX_LEN, aka sequence_max_len from training)."""
    if _session is None:
        raise HTTPException(503, "rnn_model.onnx not found -- cannot score.")
    input_name = _session.get_inputs()[0].name

    if RNN_MODE == "numeric":
        if values is None:
            raise HTTPException(400, "This RNN was trained in numeric mode -- pass "
                                      f"'values': [{RNN_WINDOW_SIZE} floats]' in the request body.")
        try:
            arr = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError):
            raise HTTPException(400, "'values' must be a flat list of numbers.")
        if arr.ndim != 1 or not np.all(np.isfinite(arr)):
            raise HTTPException(400, "'values' must be a flat list of finite numbers.")
        if arr.shape[0] != RNN_WINDOW_SIZE:
            raise HTTPException(400, f"Expected exactly {RNN_WINDOW_SIZE} values (this model's "
                                      f"training window size), got {arr.shape[0]}.")
        # [RNN REGRESSION TARGET-SCALING SERVING FIX] build_pipeline_dataloaders('RNN', values=...)
        # z-scores BOTH the windowed inputs and the target with the SAME scaler (one shared time
        # series, same units for past values and the value being forecast) -- training never saw
        # raw-scale numbers. Without scaling this request the same way, the model sees wildly
        # out-of-distribution input regardless of any output fix; the output must then be inverted
        # back with the same scaler before a caller ever sees it.
        if _rnn_target_scaler is not None and _rnn_target_scaler.get("log1p"):
            if np.any(arr <= -1):
                raise HTTPException(400, "Values must be > -1 (this model was trained on log1p(series)).")
            arr = np.log1p(arr.astype(np.float64))
        if _rnn_target_scaler is not None:
            _tmean = float(np.asarray(_rnn_target_scaler["mean_"]).reshape(-1)[0])
            _tscale = float(np.asarray(_rnn_target_scaler["scale_"]).reshape(-1)[0])
            arr = ((arr.astype(np.float64) - _tmean) / _tscale).astype(np.float32)
        x = arr.reshape(1, RNN_WINDOW_SIZE, 1)
        outputs = _session.run(None, {input_name: x})
        preds = outputs[0]
        if _rnn_target_scaler is not None:
            preds = np.asarray(preds, dtype=np.float64) * _tscale + _tmean
            if _rnn_target_scaler.get("log1p"):
                preds = np.expm1(preds)
        return {"prediction": preds.tolist()}

    if RNN_MODE == "text":
        if text is None:
            raise HTTPException(400, "This RNN was trained in text mode -- pass 'text': '...' "
                                      "in the request body.")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(422, "'text' must be a non-empty string.")
        if len(text) > _MAX_TEXT_CHARS:
            raise HTTPException(413, "'text' is too long (limit " + str(_MAX_TEXT_CHARS) + " characters).")
        if _rnn_vocab is None or _rnn_max_len is None:
            raise HTTPException(503, "rnn_vocab.json not found -- cannot tokenize raw text.")
        import re as _re_seq
        tokens = _re_seq.findall(r"\w+", text.lower())
        unk = _rnn_vocab.get("<UNK>", 1)
        ids = [_rnn_vocab.get(tok, unk) for tok in tokens[:_rnn_max_len]]
        token_ids = np.zeros((1, _rnn_max_len), dtype=np.int64)
        token_ids[0, :len(ids)] = ids
        lengths = np.array([max(len(ids), 1)], dtype=np.int64)
        length_input_name = _session.get_inputs()[1].name if len(_session.get_inputs()) > 1 else None
        feed = {input_name: token_ids}
        if length_input_name:
            feed[length_input_name] = lengths
        outputs = _session.run(None, feed)
        return _with_class_probs({"prediction": outputs[0].tolist()}, outputs[0])  # text mode = classification

    raise HTTPException(500, f"Unknown RNN_MODE '{RNN_MODE}' baked into this bundle -- expected "
                              f"'numeric' or 'text'.")


@app.post("/predict/sequence")
async def predict_sequence(req: SequenceRequest, identity: dict = Depends(check_rate_limit)):
    return await asyncio.to_thread(_score_sequence, req.values, req.text)


@app.get("/health")
def health(response: Response):
    # Deliberately NOT behind require_api_key: load balancers / k8s probes must reach it without a
    # credential. It exposes only readiness flags, never predictions or model internals.
    # Returns a real 503 when this model's artifact failed to load -- or when authentication is required but no key store is
    # available (auth_ready=false) -- so k8s's readinessProbe keeps a broken pod out of the load balancer instead of marking it Ready.
    _required_loaded = _session is not None and (RNN_MODE != 'text' or (_rnn_vocab is not None and _rnn_max_len is not None))
    _auth_ok = _auth_ready()
    if not (_required_loaded and _auth_ok):
        response.status_code = 503
    return {"status": "ok" if (_required_loaded and _auth_ok) else ("auth_not_ready" if _required_loaded else "model_not_loaded"),
            "model_type": "RNN",
            "model_loaded": _required_loaded,
            "auth_enforced": REQUIRE_AUTH,
            "auth_ready": _auth_ok,
            "airgapped": os.environ.get("AIRGAPPED_MODE", "false").lower() == "true",
            "onnx_loaded": _session is not None,
            "artifact_load_errors": (_ARTIFACT_LOAD_ERRORS if _HEALTH_VERBOSE else sorted(_ARTIFACT_LOAD_ERRORS))}


@app.get("/healthz")
def healthz():
    # LIVENESS only: the process is up and serving HTTP. Deliberately independent of model / key-store state, so a pod that is merely
    # waiting for a Secret or still loading weights is never restarted by the livenessProbe (readiness uses /health).
    return {"status": "alive"}


@app.get("/monitoring")
def monitoring(identity: dict = Depends(require_api_key)):
    """[RULE 4] Rolling latency (p50 / p95 / max over the last DRIFT_WINDOW_SIZE requests) and, for models that ship a drift baseline, the
    latest PSI evaluation. Read-only view of what the background monitor has collected; authenticated like the prediction endpoints."""
    return _monitoring_snapshot()


# ---- Startup artifact report: one line per artifact path this bundle looks for ----------------------------
def _report_artifacts():
    for _name in sorted(k for k, v in list(globals().items()) if k.endswith("_PATH") and isinstance(v, str)):
        _p = globals()[_name]
        if _p and os.path.exists(_p):
            print("[Startup] artifact %s: found -> %s" % (_name, _p))
        elif _p:
            print("[Startup] artifact %s: MISSING -> %s (named in app.py but not in the bundle)" % (_name, _p))
        else:
            print("[Startup] artifact %s: not shipped (fine when this model uses another artifact)" % _name)


_report_artifacts()

# the helper only exists in apps that serve a tabular/ANN classifier (NLP / CNN / RNN / RAG / ... apps don't carry it)
_startup_thr = globals()["_decision_threshold_cfg"]() if "_decision_threshold_cfg" in globals() else None
if _startup_thr is not None:
    print("[Startup] manual decision threshold %s (positive_index=%d) loaded from feature_recipe.json -- binary predictions "
          "use P(positive) >= this manual threshold." % (_startup_thr["threshold"], _startup_thr["positive_index"]))
elif "_threshold_required" in globals() and globals()["_threshold_required"]():
    # [STARTUP WARNING] without a threshold every binary /predict answers HTTP 500 (the automatic 0.50 default was removed on
    # purpose). /health stays green, so the first caller used to be the one to find out -- say it in the pod log at boot.
    print("[Startup] WARNING: this is a BINARY classifier but feature_recipe.json has no decision_threshold and the "
          "DECISION_THRESHOLD env var is unset -- every prediction will answer HTTP 500 until one is set "
          "(e.g. DECISION_THRESHOLD=0.35).")
