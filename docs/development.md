# Development

How to set up an environment, run the tests and the linter, what CI checks, and how to add a dependency.

## Setup

The project targets Python 3.12. From the repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt   # requirements.txt plus ruff
```

`python -m pip install -r requirements.txt` (the README's install path) still installs everything except
the linter.

Pick the requirements file for where you are installing:

| Environment | Install |
|---|---|
| Windows or macOS laptop | `pip install -r requirements-dev.txt` (the PyPI torch wheel is CPU-only there) |
| Linux without a GPU | `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu` first, then `pip install -r requirements-dev.txt` (otherwise pip fetches the multi-gigabyte CUDA build) |
| Google Colab GPU runtime | `pip install -r requirements-no-torch.txt` (keeps Colab's preinstalled CUDA torch) |
| Demo UI only | `pip install -r requirements-demo.txt` (no torch, MLflow, SQLAlchemy, API server or test tools): runs the in-memory quick demo, `python -m src.ui.app`. The reviewer page (`python -m src.api`, [reviewer-ui.md](reviewer-ui.md)) needs `requirements-no-torch.txt` or more |

The tests and the token chunker read the `distilbert/distilroberta-base` tokenizer from two caches: the
pinned revision in `.venv/hf-cache` under the repo root, and the default Hugging Face cache. Fill both,
tokenizer files only (about 3 MB, no model weights), with:

```bash
python .github/scripts/prefetch_hf_assets.py
```

The README's setup step also downloads the encoder weights, which the model experiments need but the
tests do not. In a git worktree, run the prefetch script inside the worktree: the tests look for
`.venv/hf-cache` relative to the current directory, and without it 28 tokenizer tests fail.

## Requirements files

The requirements files are the single source of truth for dependencies. They are layered with `-r`
so each version is written exactly once:

| File | Adds | Used by |
|---|---|---|
| `requirements-demo.txt` | demo UI runtime: numpy, scikit-learn, scipy, joblib, pandas, pydantic, transformers, gradio, document readers | the demo-only install (`python -m src.ui.app`; `tests/test_ui_app.py` checks it imports no SQLAlchemy, store or API code) |
| `requirements-no-torch.txt` | `-r requirements-demo.txt` plus MLflow, SQLAlchemy, the API server (FastAPI, uvicorn, python-multipart), accelerate, PyYAML, pytest and httpx (FastAPI's test client) | Colab, where torch is preinstalled |
| `requirements.txt` | `-r requirements-no-torch.txt` plus torch | the documented full install |
| `requirements-dev.txt` | `-r requirements.txt` plus ruff | developers and CI |

Every direct dependency is pinned exactly (`==`). Transitive dependencies are not locked; a full
cross-platform lock (for example with `uv`) is planned together with the package rename.

- **numpy, scikit-learn, scipy and joblib** must match the versions the released demo model was built
  with (`src/registry/demo_model.py`). Change them only together with a new model release.
- **torch** is pinned as `torch==2.14.1`, without a local build tag such as `+cpu` or `+cu128`, so any
  build of that release satisfies it: the CPU wheel CI installs from the PyTorch CPU index, the
  CPU-only PyPI wheel on Windows and macOS, or a CUDA wheel on a Linux GPU machine. Colab is the
  exception: its preinstalled torch is a different release, so installing `requirements.txt` there
  would replace it with a multi-gigabyte download. Use `requirements-no-torch.txt` on Colab instead;
  transformers and accelerate work with Colab's torch.
- **ruff** lives only in `requirements-dev.txt`, so neither the demo nor the plain full install gets it.
- **fastapi, uvicorn and python-multipart** are the API server's (`python -m src.api`, see
  [api.md](api.md)), which also serves the reviewer page ([reviewer-ui.md](reviewer-ui.md)). They sit
  in `requirements-no-torch.txt`, not the demo layer, because the API and the reviewer page also need
  SQLAlchemy for the artifact store. Gradio depends on all three as well, so their pins must stay
  inside Gradio's version ranges; `pip check` catches a mismatch.

`pyproject.toml` holds project metadata and tool configuration (ruff) and declares no dependencies,
so the requirements files cannot drift from a second list. It is not installable yet: code is
imported as `src.<module>` and `scripts.<module>` from the repo root (`pytest.ini` sets
`pythonpath = .`), and `[tool.setuptools] packages = []` keeps `pip install .` from scattering
`src/`'s subpackages into site-packages. When the package is renamed to `clausetriage/`, the
dependencies can move into `pyproject.toml` with a lock file and the requirements files can be
generated from it.

## Running the tests

From the repo root:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python -m pytest
```

In PowerShell, set the variables first: `$env:HF_HUB_OFFLINE=1; $env:TOKENIZERS_PARALLELISM="false"`.
`HF_HUB_OFFLINE=1` makes a missing tokenizer fail fast instead of downloading it. The suite runs in
under a minute on a laptop CPU and never reads `data/cuad/test.json`.

On Windows, the three tests in `tests/test_gwen_adapter.py` that check the SHA-256 of an archived
source file fail if git converted that file's line endings to CRLF on checkout. They pass on Linux
and in CI.

## Running the linter

```bash
ruff check .
```

Ruff only checks; there is no formatter and no import sorting. The configuration is in
`pyproject.toml`:

- **Rules:** `E4`, `E7`, `E9` (ruff's defaults: import placement, comparisons, bare `except`, syntax
  errors), `F` (pyflakes: undefined names, unused imports and variables, redefinitions), `B`
  (flake8-bugbear: mutable default arguments, closures over loop variables and similar), `W605`
  (invalid escape sequences), `PLE` (pylint errors) and `T10` (leftover debugger calls).
  `B905` (`zip()` without `strict=`) is off for now: it fires 35 times on existing code.
- **Line length** is recorded as 120, the length the code already uses. It is not enforced (`E501`
  is not selected).
- **Notebooks** (`.ipynb`) are checked too.
- **Excluded:** `.claude/` (agent worktrees) and ruff's defaults (`.venv`, `venv`, `.git`, caches).
  `.gitignore` is not used for exclusion, because its `models/` rule would also hide `src/models/`.
- **Known findings** in existing code are suppressed per file and rule under
  `[tool.ruff.lint.per-file-ignores]`, not fixed, because other branches were editing those files.
  Each should be fixed and its entry deleted:

| File | Rule | Finding |
|---|---|---|
| `notebooks/baseline_test.ipynb` | F401 | unused import `pred_tfidf_baseline` |
| `scripts/evaluate_encoder_nested.py` | F401 | unused import `pandas` |
| `scripts/repair_per_category_aupr.py` | B023 | lambda in a loop reads the loop variable `aupr` (safe today: it is called within the same iteration) |
| `scripts/train_category_query.py` | F401 | unused import `pandas` |
| `scripts/train_contract_graph.py` | F401, F811 | `ContractGraph` imported at the top, unused, then re-imported inside a function |
| `scripts/train_long_context.py` | F401 | unused import `pandas` |
| `scripts/train_set_prediction.py` | F401 | unused imports `Contract`, `to_model_inputs` |
| `scripts/train_sparse_long_context.py` | F841 | `amp_enabled` assigned but never used (dead code: the autocast call computes the same condition inline) |
| `src/chunking/token_chunker.py` | F401, F811 | stray `from pydoc import text`, shadowed by every `text` parameter |
| `src/inference.py` | F401 | unused imports `GroundedEvidence`, `Unknown` |
| `src/models/encoder/long_context.py` | F401 | unused import `typing.Any` |
| `src/models/encoder/set_prediction.py` | F401 | unused import `AnswerSpan` |
| `src/models/encoder/sparse_long_context.py` | F401 | unused import `torch.nn.functional` |
| `src/models/encoder/train.py` | B028 | `warnings.warn` without `stacklevel` |
| `src/store/artifact_store.py` | F401 | unused import `typing.Any` |
| `tests/test_demo_model.py` | B028 | `warnings.warn` without `stacklevel` |
| `tests/test_retrieval_reranker.py` | F401 | unused import `PairExample` |

`gr.Progress()` as a default argument is Gradio's documented idiom, so it is marked immutable for
`B008` instead of being ignored.

## What CI checks

`.github/workflows/ci.yml` runs on every pull request and every push to `main`, on Ubuntu 24.04 with
Python 3.12, in two parallel jobs:

1. **Ruff:** installs only the ruff version pinned in `requirements-dev.txt` and runs
   `ruff check .`, with findings shown as annotations on the PR.
2. **Pytest:** installs the CPU build of torch from the PyTorch CPU index (the version is read from
   `requirements.txt`), then `requirements-dev.txt`, runs `pip check` and asserts torch has no CUDA.
   It restores the cached tokenizer files, runs `.github/scripts/prefetch_hf_assets.py` to fetch any
   that are missing, and runs `pytest` with `HF_HUB_OFFLINE=1`.

pip downloads are cached by the hash of the requirements files. Expect about 5 minutes for a cold
run (mostly downloading and installing torch, MLflow, gradio and the scientific stack) and about
2 to 3 minutes once the pip cache is warm; the tests themselves take about a minute.

## Adding a dependency

1. Install it into your venv and note the exact version (`pip show <package>`).
2. Add `<package>==<version>` to the lowest layer that needs it: `requirements-demo.txt` if the demo
   UI (`src/ui/app.py`, `src/inference.py` and what they import) uses it, `requirements-no-torch.txt` for
   training, tracking or test tools, `requirements-dev.txt` for development-only tools. Only torch
   belongs in `requirements.txt` itself.
3. Keep one requirement per line with no trailing comment on the `torch==` and `ruff==` lines: CI
   reads those two lines with `grep`.
4. Check the version has wheels for Linux and Windows on Python 3.12, then run `pip check`,
   `ruff check .` and the tests. CI installs from scratch, so a missing entry shows up there as an
   `ImportError`.
5. Do not add dependencies to `pyproject.toml`.
