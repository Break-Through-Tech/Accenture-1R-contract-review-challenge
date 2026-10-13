import pytest
from src.chunking.chunk import Chunk
from src.labels.categories import CATEGORIES, CATEGORY_TO_INDEX
from src.labels.projection import LabeledChunk
from src.models.encoder.config import TrainConfig
from scripts.smoke_train import (
    CHUNKS_PER_GROUP,
    PASS_THRESHOLD,
    SMOKE_OUTPUT_DIR,
    main,
    make_smoke_config,
    make_smoke_subset,
)


def make_config(**overrides) -> TrainConfig:
    """Build a valid TrainConfig that differs from the smoke settings, with any fields overridden."""
    values = {
        "model_name": "distilbert/distilroberta-base",
        "max_length": 512,
        "window_chars": 800,
        "overlap_chars": 200,
        "val_fraction": 0.2,
        "seed": 42,
        "lr": 3.0e-5,
        "epochs": 4,
        "batch_size": 16,
        "grad_accum": 2,
        "fp16": True,
        "pos_weight_cap": 50.0,
        "output_dir": "outputs/encoder",
        "mlflow_experiment": "test-experiment",
    }
    values.update(overrides)
    return TrainConfig(**values)


def make_chunks(num_labeled: int, num_unlabeled: int) -> list[LabeledChunk]:
    """Build unlabeled and Parties-labeled chunks with unique contract ids c0, c1, ... in that mixed order."""
    chunks = []
    for i in range(num_labeled + num_unlabeled):
        labels = [0] * len(CATEGORIES)
        if i % 2 == 0 and i // 2 < num_labeled:
            labels[CATEGORY_TO_INDEX["Parties"]] = 1
        text = f"clause {i}"
        chunk = Chunk(contract_id=f"c{i}", start=0, end=len(text), text=text, chunk_index=0)
        chunks.append(LabeledChunk(chunk=chunk, labels=labels))
    return chunks


# ----- make_smoke_subset: sampling -----
def test_make_smoke_subset_takes_per_group_of_each_kind_labeled_first():
    """Checks the subset has 3 labeled then 3 unlabeled chunks when there are plenty of each."""
    subset = make_smoke_subset(make_chunks(10, 30), per_group=3, seed=0)

    assert len(subset) == 6
    assert all(any(c.labels) for c in subset[:3])
    assert not any(any(c.labels) for c in subset[3:])


def test_make_smoke_subset_group_smaller_than_per_group_returns_what_exists():
    """Checks a group with fewer chunks than per_group is returned whole instead of raising."""
    subset = make_smoke_subset(make_chunks(2, 30), per_group=5, seed=0)

    assert sum(any(c.labels) for c in subset) == 2
    assert sum(not any(c.labels) for c in subset) == 5


def test_make_smoke_subset_without_any_labeled_chunks_returns_only_unlabeled():
    """Checks the helper still works when the input has no labeled chunks at all."""
    subset = make_smoke_subset(make_chunks(0, 10), per_group=4, seed=0)

    assert len(subset) == 4
    assert not any(any(c.labels) for c in subset)


def test_make_smoke_subset_same_seed_gives_the_same_subset():
    """Checks the sample is reproducible for a given seed."""
    chunks = make_chunks(20, 40)

    assert make_smoke_subset(chunks, 5, seed=7) == make_smoke_subset(chunks, 5, seed=7)


def test_make_smoke_subset_different_seeds_give_different_samples():
    """Checks the seed matters, so the subset is a sample and not just the first chunks."""
    chunks = make_chunks(20, 40)

    assert make_smoke_subset(chunks, 5, seed=1) != make_smoke_subset(chunks, 5, seed=2)


def test_make_smoke_subset_chunks_come_from_the_input_without_duplicates():
    """Checks every sampled chunk is one of the inputs and none is repeated."""
    chunks = make_chunks(10, 30)

    subset = make_smoke_subset(chunks, per_group=4, seed=3)

    ids = [c.chunk.contract_id for c in subset]
    assert len(set(ids)) == len(ids)
    assert set(ids) <= {c.chunk.contract_id for c in chunks}


# ----- make_smoke_config: overrides -----
def test_make_smoke_config_applies_the_cpu_friendly_overrides():
    """Checks the smoke config trains fast on a CPU: no fp16, short sequences, small batch, smoke folder."""
    smoke = make_smoke_config(make_config())

    assert smoke.fp16 is False
    assert smoke.max_length == 128
    assert smoke.batch_size == 8
    assert smoke.grad_accum == 1
    assert smoke.output_dir == SMOKE_OUTPUT_DIR
    assert smoke.lr > make_config().lr
    assert smoke.epochs > make_config().epochs


def test_make_smoke_config_keeps_the_fields_it_does_not_override():
    """Checks model name, chunking sizes, seed and weight cap are carried over unchanged."""
    original = make_config(seed=7, pos_weight_cap=25.0)

    smoke = make_smoke_config(original)

    assert smoke.model_name == original.model_name
    assert smoke.window_chars == original.window_chars
    assert smoke.overlap_chars == original.overlap_chars
    assert smoke.seed == 7
    assert smoke.pos_weight_cap == 25.0


def test_make_smoke_config_leaves_the_original_config_unchanged():
    """Checks the original config is not modified, since a real run's settings must stay intact."""
    original = make_config()

    make_smoke_config(original)

    assert original == make_config()
    assert original.fp16 is True


def test_make_smoke_config_returns_a_train_config():
    """Checks the copy is still a TrainConfig, so fit accepts it."""
    assert isinstance(make_smoke_config(make_config()), TrainConfig)


# ----- main: pass/fail exit code and wiring -----
@pytest.fixture
def fake_pipeline(monkeypatch):
    """Replace load_data and fit with fakes and return a dict that records what fit received."""
    record = {"metrics": {"eval_macro_aupr": 0.95}}
    monkeypatch.setattr("scripts.smoke_train.load_data", lambda config: (make_chunks(50, 100), []))

    def fake_fit(config, train_chunks, val_chunks):
        """Record fit's arguments and return the metrics the test configured."""
        record.update(config=config, train=train_chunks, val=val_chunks)
        return record["metrics"]

    monkeypatch.setattr("scripts.smoke_train.fit", fake_fit)
    return record


@pytest.mark.parametrize("macro_aupr, expected_code", [
    (0.99, 0),                    # memorized the subset
    (PASS_THRESHOLD, 0),          # exactly on the threshold passes
    (PASS_THRESHOLD - 0.01, 1),   # just below fails
    (0.2, 1),                     # not learning at all
])
def test_main_exit_code_reflects_whether_macro_aupr_reaches_the_threshold(fake_pipeline, macro_aupr, expected_code):
    """Checks main returns 0 only when eval_macro_aupr is at least PASS_THRESHOLD."""
    fake_pipeline["metrics"] = {"eval_macro_aupr": macro_aupr}

    assert main([]) == expected_code


def test_main_prints_pass_or_fail_with_the_score(fake_pipeline, capsys):
    """Checks the verdict line names PASS or FAIL and the macro AUPR."""
    fake_pipeline["metrics"] = {"eval_macro_aupr": 0.5}

    main([])

    out = capsys.readouterr().out
    assert "SMOKE FAIL" in out
    assert "0.5000" in out


def test_main_trains_and_evaluates_on_the_same_small_subset_with_the_smoke_config(fake_pipeline):
    """Checks fit gets the smoke config and one subset of at most 2 * CHUNKS_PER_GROUP chunks, used for both splits."""
    main([])

    assert fake_pipeline["config"].fp16 is False
    assert fake_pipeline["config"].output_dir == SMOKE_OUTPUT_DIR
    assert fake_pipeline["train"] is fake_pipeline["val"]
    assert len(fake_pipeline["train"]) == 2 * CHUNKS_PER_GROUP
