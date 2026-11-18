# Docker

The `Dockerfile` and `compose.yaml` at the repo root run the API server with the reviewer page
(`python -m clausetriage.api`, see [api.md](api.md) and [reviewer-ui.md](reviewer-ui.md)), so the
project can be tried with only Docker installed. Scores use the team-set `v2-team-proposed` weights,
which no legal expert has validated, and are not legal advice.

## Running it

From the repo root, with Docker and Docker Compose:

```bash
docker compose up --build          # first build: a few minutes
```

Then open http://127.0.0.1:8000/ui/ for the reviewer page, or http://127.0.0.1:8000/docs for the API.
The server log in that terminal prints `Uvicorn running on http://0.0.0.0:8000` and a warning that
the API has no authentication; both are expected inside the container (see the next section). Use
`docker compose up -d --build` to run it in the background and `docker compose logs -f` to follow
the log.

| To | Run |
|---|---|
| Stop the server, keep the data | `docker compose down` |
| Start it again | `docker compose up -d` |
| Rebuild after pulling new code (the data is kept) | `docker compose up -d --build` |
| Delete the container, the image and all stored data | `docker compose down -v --rmi all` |

## Localhost only

The API has no authentication: anyone who can reach the port can upload files, read every stored
contract and record verdicts. The project rule is that it listens on this machine only. Inside a
container the server has to listen on all interfaces (`--host 0.0.0.0`), or Docker could not forward
the port to it, so the server prints its no-authentication warning. The protection is in
`compose.yaml` instead, which publishes the port on the host's loopback address only:

```yaml
ports:
  - "127.0.0.1:8000:8000"
```

Do not change this to `"8000:8000"`: that listens on every network interface of the host, and Docker's
port forwarding can bypass the host firewall on Linux. To use another host port, change only the first
number, for example `"127.0.0.1:8080:8000"`. If you run the image without Compose, publish it the same
way: `docker run -p 127.0.0.1:8000:8000 -v clausetriage-data:/data clausetriage:local`.

## What is in the image

- **No PyTorch.** The served model is word and character TF-IDF, so the image installs
  `requirements-server.txt`, the smallest requirements layer that runs the server: the demo runtime
  plus SQLAlchemy, FastAPI, uvicorn and python-multipart. MLflow, torch and the test tools are not
  installed. On 2026-11-18 the image was 1.24 GB on disk (297 MB compressed), most of it the Python
  packages (gradio, transformers, pandas, scipy, scikit-learn).
- **The package**, installed editable in `/app`, and the files it reads at run time:
  `data/cuad/category_descriptions.csv`, the sample PDFs in `data/demo_contracts/`, the calibrated
  thresholds, and `data/cuad/train_separate_questions.json` (for the local model build and for loading
  the development contracts). `.dockerignore` is an allowlist, so nothing else enters the build. In
  particular `data/cuad/test.json`, `CUADv1.json` and `cuad-v1.zip`, which hold the test contracts,
  are never sent to Docker.
- **The tokenizer:** the build downloads the pinned DistilRoBERTa tokenizer files (about 3 MB, no
  model weights) from Hugging Face into `/app/.venv/hf-cache`, so building needs network access.
  The container runs with `HF_HUB_OFFLINE=1` and `GRADIO_ANALYTICS_ENABLED=False`, and needs no
  network once built.
- **The model,** in `/app/models/word_char_tfidf.joblib`, taken at build time from the first of:
  1. `models/word_char_tfidf.joblib` in your checkout, if you have it (it is git-ignored; the quick
     demo or the server saves it there on first run, or download it with
     `gh release download model-word-char-tfidf-v1 -p word_char_tfidf.joblib -D models`);
  2. the GitHub Release `model-word-char-tfidf-v1`, kept only if its SHA-256 matches `BUNDLE_SHA256`;
  3. a local build from the training file (`build_demo_bundle()`, about 3 minutes during the build).

  The build log prints the file's SHA-256 and whether it is the released file. Only the released file
  (`6fd84c28...`) gives the model version `word_char_tfidf@sha256:6fd84c28600f` that the evaluated
  results and other teammates' databases use. A local build is a different file, so its results are
  stored under a different model version.
- The server runs as an unprivileged user (`clausetriage`, uid 1000), with a health check on
  `/health`.

## The database

The SQLite database (scored contracts, weight versions, review feedback) is
`/data/clausetriage.sqlite` in the named volume `clausetriage-data`. It survives `docker compose down`,
restarts and rebuilds, and is deleted only by `docker compose down -v`. Run one server process per
database, as outside Docker.

To score CUAD development contracts into it, stop the server first, then run the loader in a one-off
container on the same volume:

```bash
docker compose stop
docker compose run --rm clausetriage python -m scripts.load_dataset_contracts --database /data/clausetriage.sqlite --limit 10
docker compose up -d
```

In Git Bash on Windows, set `MSYS_NO_PATHCONV=1` first, or Git Bash rewrites `/data/...` into a
Windows path.

## What was checked

On 2026-11-18, with Docker 29.6.1 on Windows 11 (Linux containers):

- `docker compose up --build` built the image from a checkout that had the released model file; the
  model in the image has SHA-256 `6fd84c28...` and the image has no torch, MLflow or pytest.
- `/health` answered with model `word_char_tfidf@sha256:6fd84c28600f` and weights
  `v2-team-proposed`; the container reported healthy.
- Uploading `data/demo_contracts/nasa_kbr_agreement.pdf` to `/contracts/upload` scored it, and its
  `register.csv` and `register.json` were byte-identical to those from the server outside Docker.
- `/ui/` and `/docs` answered 200, and the reviewer page's start handler, called through
  `gradio_client`, listed the upload.
- Port 8000 answered on 127.0.0.1 and refused connections on the machine's network addresses.
- After `docker compose down`, a rebuild and `docker compose up`, the scored contract was still there.
- `scripts.load_dataset_contracts --limit 3` scored three contracts into the volume.
- The local build fallback, run in the image with an unreachable download URL, built a working model
  in about 3 minutes.

Not checked: Linux and macOS hosts, ARM machines, rootless Docker, and building without a network.
