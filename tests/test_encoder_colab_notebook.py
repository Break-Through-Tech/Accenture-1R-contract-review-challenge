import ast
import json
import re
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "encoder_colab.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    """Load the Colab notebook as plain JSON (nbformat is not a project dependency)."""
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells(notebook: dict) -> list[str]:
    """Return the source of every code cell, in order."""
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def first_cell_containing(notebook: dict, text: str) -> int:
    """Index (among code cells) of the first cell whose source contains text."""
    return next(index for index, source in enumerate(code_cells(notebook)) if text in source)


# ----- structure -----
def test_notebook_is_nbformat_4_with_no_saved_outputs(notebook):
    """Checks the file is an nbformat 4 notebook committed without outputs or execution counts."""
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_notebook_code_cells_are_valid_python(notebook):
    """Checks every code cell parses once IPython shell lines (!cmd) are set aside."""
    for source in code_cells(notebook):
        ast.parse(re.sub(r"(?m)^(\s*)!.*$", r"\1pass", source))


# ----- order: each step only uses what earlier cells set up -----
@pytest.mark.parametrize("earlier, later", [
    ("REPO_DIR =", "torch.cuda.is_available()"),
    ("torch.cuda.is_available()", "git\", \"clone\""),
    ("RESULTS_ROOT =", "git\", \"clone\""),
    ("git\", \"clone\"", "pip\", \"install\""),
    ("pip\", \"install\"", "scripts.evaluate_encoder_nested"),
    ("scripts.evaluate_encoder_nested", "scripts.compare_encoder_tfidf"),
    ("scripts.compare_encoder_tfidf", "zipfile.ZipFile"),
])
def test_notebook_steps_run_in_dependency_order(notebook, earlier, later):
    """Checks settings, GPU check, clone, install, training, comparison and packaging come in that order."""
    assert first_cell_containing(notebook, earlier) < first_cell_containing(notebook, later)


# ----- safety -----
def test_notebook_contains_no_hard_coded_token(notebook):
    """Checks no GitHub token pattern appears and the token comes from a Colab secret or a hidden prompt."""
    raw = NOTEBOOK.read_text(encoding="utf-8")
    code = "\n".join(code_cells(notebook))

    assert not re.search(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}", raw)
    assert 'userdata.get("GITHUB_TOKEN")' in code and "getpass.getpass" in code


def test_notebook_install_holds_preinstalled_torch(notebook):
    """Checks pip installs requirements with a constraint pinning the preinstalled CUDA torch build."""
    install = code_cells(notebook)[first_cell_containing(notebook, "pip\", \"install\"")]

    assert 'f"torch=={torch.__version__}' in install
    assert '"-c", str(constraint)' in install


def test_notebook_never_reads_the_test_split(notebook):
    """Checks the notebook never mentions test.json in code."""
    assert all("test.json" not in source for source in code_cells(notebook))
