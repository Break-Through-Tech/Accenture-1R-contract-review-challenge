# Development

How to set up an environment, run the tests and the linter, what CI checks, how the package is
installed, and how to add a dependency.

## Setup

The project targets Python 3.12. From the repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt   # requirements.txt plus ruff
python -m pip install -e .                      # the clausetriage package, editable
```

`python -m pip install -r requirements.txt` (the README's install path) still installs everything except
the linter.

Pick the requirements file for where you are installing:

| Environment | Install |
|---|---|
| Windows or macOS laptop | `pip install -r requirements-dev.txt` (the PyPI torch wheel is CPU-only there) |
| Linux without a GPU | `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu` first, then `pip install -r requirements-dev.txt` (otherwise pip fetches the multi-gigabyte CUDA build) |
| Google Colab GPU runtime | `pip install -r requirements-no-torch.txt` (keeps Colab's preinstalled CUDA torch) |
| Demo UI only | `pip install -r requirements-demo.txt` (no torch, MLflow, SQLAlchemy, API server or test tools): runs the in-memory quick demo, `python -m clausetriage.ui.app` |
| API server and reviewer page only | `pip install -r requirements-server.txt` (no torch, MLflow or test tools): runs `python -m clausetriage.api` ([api.md](api.md), [reviewer-ui.md](reviewer-ui.md)). The Docker image installs this layer ([docker.md](docker.md)) |

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
| `requirements-demo.txt` | demo UI runtime: numpy, scikit-learn, scipy, joblib, pandas, pydantic, transformers, gradio, document readers | the demo-only install (`python -m clausetriage.ui.app`; `tests/test_ui_app.py` checks it imports no SQLAlchemy, store or API code), and the package's own dependencies in `pyproject.toml` |
| `requirements-server.txt` | `-r requirements-demo.txt` plus SQLAlchemy and the API server (FastAPI, uvicorn, python-multipart) | the server-only install and the Docker image |
| `requirements-no-torch.txt` | `-r requirements-server.txt` plus MLflow, accelerate, PyYAML, pytest and httpx (FastAPI's test client) | Colab, where torch is preinstalled |
| `requirements.txt` | `-r requirements-no-torch.txt` plus torch | the documented full install |
| `requirements-dev.txt` | `-r requirements.txt` plus ruff | developers and CI |

Every direct dependency is pinned exactly (`==`). Transitive dependencies are not locked; a full
cross-platform lock (for example with `uv`) is planned together with the package rename.

- **numpy, scikit-learn, scipy and joblib** must match the versions the released demo model was built
  with (`clausetriage/registry/demo_model.py`). Change them only together with a new model release.
- **torch** is pinned as `torch==2.14.1`, without a local build tag such as `+cpu` or `+cu128`, so any
  build of that release satisfies it: the CPU wheel CI installs from the PyTorch CPU index, the
  CPU-only PyPI wheel on Windows and macOS, or a CUDA wheel on a Linux GPU machine. Colab is the
  exception: its preinstalled torch is a different release, so installing `requirements.txt` there
  would replace it with a multi-gigabyte download. Use `requirements-no-torch.txt` on Colab instead;
  transformers and accelerate work with Colab's torch.
- **ruff** lives only in `requirements-dev.txt`, so neither the demo nor the plain full install gets it.
- **fastapi, uvicorn and python-multipart** are the API server's (`python -m clausetriage.api`, see
  [api.md](api.md)), which also serves the reviewer page ([reviewer-ui.md](reviewer-ui.md)). They sit
  in `requirements-server.txt`, not the demo layer, because the API and the reviewer page also need
  SQLAlchemy for the artifact store. Gradio depends on all three as well, so their pins must stay
  inside Gradio's version ranges; `pip check` catches a mismatch.

`pyproject.toml` holds project metadata and tool configuration (ruff). It pins no dependencies
itself: `[tool.setuptools.dynamic]` reads the package's dependencies from `requirements-demo.txt`, so
the requirements files stay the single source of truth. Keep that file to plain `package==version`
lines and comments, because setuptools cannot read `-r` or other pip options. There is no
cross-platform lock file.

## The clausetriage package

All pipeline code is the `clausetriage` package in `clausetriage/`. `scripts/` and `tests/` are not
part of it: run them from the repo root (`python -m scripts.<name>`, `python -m pytest`), where
`pytest.ini`'s `pythonpath = .` puts the repo root on the path.

- **Install it editable:** `pip install -e .` installs the package and the demo runtime
  (`requirements-demo.txt`) and makes `import clausetriage` work from any folder, for example in
  notebooks. The code reads `data/`, `models/`, `results/` and `.venv/hf-cache` relative to the
  checkout, so a non-editable `pip install .` would look for them inside site-packages. The weight
  sets in `clausetriage/risk/weights/` are package data. CI installs the package editable and
  imports it from outside the checkout.
- **Renamed from `src`:** until November 2026 the package was called `src`. Saved results under
  `results/`, `docs/final-test-evaluation.md` (a registered record) and the git history still name
  `src/` paths; they are left as they were. After pulling the rename, delete any leftover `src/`
  folder in your checkout (it holds only ignored `__pycache__` files).
- **Pickles from before the rename:** a pickle records the module path of every class in it, and the
  released model file (`models/word_char_tfidf.joblib`) names `src.registry.bundle` and
  `src.registry.predictors`. That file is the frozen artifact of the sealed test evaluation and is
  pinned by SHA-256, so it is never rebuilt or re-saved. `clausetriage/legacy.py` resolves the old
  names instead: inside `legacy_src_imports()`, importing `src.<module>` returns the
  `clausetriage.<module>` object itself, and `load_joblib()` loads a pickle inside that block. Every
  bundle load (`demo_model.py`, `mlflow_registry.py`, the sealed evaluation script, the experiment
  scripts) goes through `load_joblib()`, and the byte-pinned PR #48 snapshot
  (`experiments/archive/pr48_token_chunking.py`, which imports `src.chunking.chunk`) is executed
  inside the block. Outside it, `import src` still fails, so a stale import in the code is an error.
  `tests/test_legacy.py` covers the aliasing, a pickle written with the old paths, and, when the
  file is present, the released model itself.
- **Databases from before the rename** open unchanged: the artifact store keys model versions by
  the bundle file's SHA-256 and weight versions by their content, never by module path.

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
  `B905` (`zip()` without `strict=`) stays off: it fires 47 times in 30 files. Twelve of those are
  pairwise loops such as `zip(chunks, chunks[1:])` that truncate on purpose, where the automatic fix
  (`strict=True`) would raise, and the rest include serving code and the sealed evaluation script.
  Enable it file by file, checking each call.
- **Line length** is recorded as 120, the length the code already uses. It is not enforced (`E501`
  is not selected).
- **Notebooks** (`.ipynb`) are checked too.
- **Excluded:** `.claude/` (agent worktrees) and ruff's defaults (`.venv`, `venv`, `.git`, caches).
  `.gitignore` is not used for exclusion, because its `models/` rule would also hide `clausetriage/models/`.
- **No per-file suppressions.** The findings that were suppressed per file while parallel branches
  were open have been fixed, and `[tool.ruff.lint.per-file-ignores]` is gone. Fix new findings
  rather than suppressing them; if a pattern is deliberate, use a narrow `# noqa: <rule>` with a
  comment saying why (as `.github/scripts/prefetch_hf_assets.py` does for an import after its
  `sys.path` setup).

`gr.Progress()` as a default argument is Gradio's documented idiom, so it is marked immutable for
`B008` instead of being ignored.

## What CI checks

`.github/workflows/ci.yml` runs on every pull request and every push to `main`, on Ubuntu 24.04 with
Python 3.12, in two parallel jobs:

1. **Ruff:** installs only the ruff version pinned in `requirements-dev.txt` and runs
   `ruff check .`, with findings shown as annotations on the PR.
2. **Pytest:** installs the CPU build of torch from the PyTorch CPU index (the version is read from
   `requirements.txt`), then `requirements-dev.txt` and the package (`pip install -e .`), runs
   `pip check`, asserts torch has no CUDA, and imports the package from outside the checkout.
   It restores the cached tokenizer files, runs `.github/scripts/prefetch_hf_assets.py` to fetch any
   that are missing, and runs `pytest` with `HF_HUB_OFFLINE=1`.

pip downloads are cached by the hash of the requirements files. Expect about 5 minutes for a cold
run (mostly downloading and installing torch, MLflow, gradio and the scientific stack) and about
2 to 3 minutes once the pip cache is warm; the tests themselves take about a minute.

## Adding a dependency

1. Install it into your venv and note the exact version (`pip show <package>`).
2. Add `<package>==<version>` to the lowest layer that needs it: `requirements-demo.txt` if the demo
   UI (`clausetriage/ui/app.py`, `clausetriage/inference.py` and what they import) uses it,
   `requirements-server.txt` if only the API server or the store needs it, `requirements-no-torch.txt`
   for training, tracking or test tools, `requirements-dev.txt` for development-only tools. Only torch
   belongs in `requirements.txt` itself.
3. Keep one requirement per line with no trailing comment on the `torch==` and `ruff==` lines: CI
   reads those two lines with `grep`.
4. Check the version has wheels for Linux and Windows on Python 3.12, then run `pip check`,
   `ruff check .` and the tests. CI installs from scratch, so a missing entry shows up there as an
   `ImportError`.
5. Do not write dependencies into `pyproject.toml`; it reads them from `requirements-demo.txt`.
