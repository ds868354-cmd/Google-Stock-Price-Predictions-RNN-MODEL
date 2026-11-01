# [REPRODUCIBLE BASE] the base image is a build ARG so a deployer can PIN IT BY DIGEST without editing this file:
#   docker build --build-arg PYTHON_IMAGE=python:3.11-slim@sha256:<digest> .
# (a moving tag such as python:3.11-slim changes under you between builds; ci.yml's Trivy scan flags the CVEs of whatever it resolves to).
ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE}

# [DOCKERFILE PRODUCTION FIX] real-time logs (no stdout buffering) and no .pyc clutter in the image layer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# [DOCKERFILE PRODUCTION FIX] libgomp1 (OpenMP runtime) is required at import time by ONNX Runtime and LightGBM;
# python:*-slim does not ship it, so the app crashed on startup with "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# [NON-ROOT FIX] the app user is created BEFORE any COPY so files are owned by it directly (COPY --chown). The old trailing
# recursive chown rewrote every copied file into a second layer and roughly doubled the image size.
RUN useradd --create-home --uid 10001 appuser && chown appuser:appuser /app

# [AUTH SECURE-DEFAULT] with APP_ENV=production a missing / empty api_keys.json means "refuse requests (503)", never "serve
# open". Local container without keys: docker run -e APP_ENV=staging -e REQUIRE_AUTH=false ...
ENV APP_ENV=production

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# [RULE 1 -- DYNAMIC ARTIFACT COPY] No file names are hard-coded here. The build context is filtered by .dockerignore, a DEFAULT-DENY
# allow-list: only app.py, requirements.txt, the safe model / preprocessing formats (*.onnx, *.onnx.data, *.data, *.safetensors, *.json,
# *.bin, *.pth, *.pt, *.h5, *.pb) and the artifact folders (artifacts/, llm_lora_adapter/, llm_base_model/, embedding_model/) get through;
# credentials (api_keys.json, .env, keys), run metadata and every pickle / joblib file (*.pkl, *.pickle, *.joblib -- forbidden formats)
# are excluded wherever they sit. So ONE instruction copies exactly the artifacts that exist when `docker build` runs: a model file added
# after this Dockerfile was generated is still copied, and /health cannot answer 503 because a file was missing from a hand-made list.
# Credentials are NEVER in a layer: mount api_keys.json at run time (k8s Secret volume / `docker run -v`, then API_KEYS_PATH) or inject
# the same JSON as the API_KEYS_JSON environment variable.
COPY --chown=appuser:appuser . ./

# [PORT FIX] ONE source of truth: PORT is used by EXPOSE, HEALTHCHECK and uvicorn below, so the exposed port
# can never differ from the port the server actually binds. k8s.yaml sets the same PORT for the container.
ENV PORT=8000
EXPOSE $PORT

# [NON-ROOT FIX] do not run the model server as root: a container escape / RCE in the API would otherwise own the
# container. appuser owns /app (created above); on a read-only root filesystem (k8s.yaml) the ONNX external-data repair stages the
# files in TMPDIR instead of writing next to the model.
USER appuser

# NOTE: this container does NOT block outbound network egress at the OS/network level.
# app.py only self-REPORTS an "airgapped" flag on /health (from the AIRGAPPED_MODE env var,
# see app.py's health handler) -- that flag is informational, not an enforced firewall. If you
# need actual network isolation, apply it at the platform level (e.g. a NetworkPolicy in
# k8s.yaml, or --network=none for local `docker run`). This HEALTHCHECK below is a loopback
# call regardless, so it works either way.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'))" || exit 1

# --host 0.0.0.0 is mandatory in a container (127.0.0.1 would be unreachable from outside); `exec` keeps uvicorn as
# PID 1 so SIGTERM from docker/k8s shuts it down gracefully.
# --no-server-header: do not advertise the uvicorn version to scanners; --timeout-keep-alive 5 frees idle connections quickly (slow-client DoS).
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port $PORT --no-server-header --timeout-keep-alive 5"]
