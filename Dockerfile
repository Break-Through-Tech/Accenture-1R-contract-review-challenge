# syntax=docker/dockerfile:1
# The ClauseTriage API server with the reviewer page (python -m clausetriage.api), for trying the
# project with only Docker installed. See docs/docker.md. Build and run with: docker compose up --build
#
# The image holds the TF-IDF model and the pinned tokenizer, so it starts without network access.
# It has no PyTorch: it installs requirements-server.txt, the smallest layer that runs the API.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Dependencies first, so a code change doesn't reinstall them
COPY requirements-demo.txt requirements-server.txt ./
RUN pip install -r requirements-server.txt

# The package, the scripts (for loading the CUAD development contracts) and the files the code reads
# at run time. .dockerignore keeps everything else out, including the test contracts
COPY pyproject.toml README.md ./
COPY clausetriage/ clausetriage/
COPY scripts/ scripts/
COPY .github/scripts/prefetch_hf_assets.py .github/scripts/
COPY data/cuad/category_descriptions.csv data/cuad/train_separate_questions.json data/cuad/
COPY data/demo_contracts/ data/demo_contracts/
COPY results/word_char_tfidf_nested/word_char_tfidf_thresholds.json results/word_char_tfidf_nested/
RUN pip install --no-deps -e .

# The pinned tokenizer files (about 3 MB, no model weights) into .venv/hf-cache, where the code looks
RUN python -c "import sys; sys.path.insert(0, '.github/scripts'); \
from prefetch_hf_assets import PINNED_CACHE, TOKENIZER_REVISION, prefetch; \
print('tokenizer', prefetch(TOKENIZER_REVISION, PINNED_CACHE))"
ENV HF_HUB_OFFLINE=1

# The model: models/word_char_tfidf.joblib from the build context if it is there, else the GitHub
# Release download (kept only if its SHA-256 matches), else a local build from the training file
RUN --mount=type=bind,source=.,target=/context \
    if [ -f /context/models/word_char_tfidf.joblib ]; then \
        mkdir -p models && cp /context/models/word_char_tfidf.joblib models/; \
    fi \
 && python -c "from clausetriage.registry.demo_model import BUNDLE_PATH, BUNDLE_SHA256, file_sha256, \
load_or_build_bundle; load_or_build_bundle(); sha = file_sha256(BUNDLE_PATH); \
print('model', sha, 'released file' if sha == BUNDLE_SHA256 else 'built locally, not the released file')"

# Run as an unprivileged user. The database lives in /data, a volume (see compose.yaml)
RUN useradd --create-home --uid 1000 clausetriage \
 && mkdir -p /data \
 && chown clausetriage:clausetriage /data
USER clausetriage
VOLUME /data

# Inside the container the server must listen on all interfaces to be reachable through the published
# port, so it prints its no-authentication warning. compose.yaml publishes the port on the host's
# 127.0.0.1 only, which keeps the project's localhost-only rule
EXPOSE 8000
# No Gradio usage analytics or version check calls from the container
ENV GRADIO_ANALYTICS_ENABLED=False
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["python", "-m", "clausetriage.api", "--host", "0.0.0.0", "--port", "8000", "--database", "/data/clausetriage.sqlite"]
