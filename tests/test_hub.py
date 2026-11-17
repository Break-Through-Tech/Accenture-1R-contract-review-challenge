from pathlib import Path

import pytest

from src.models.encoder import hub
from src.models.encoder.hub import (
    CACHE_ENV,
    HF_OFFLINE_ENV,
    OFFLINE_ENV,
    HubOptions,
    export_hub_options,
    load_pinned_tokenizer,
    parse_flag,
    resolve_hub_options,
)

PINNED = "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b"


# ----- parse_flag: environment booleans -----
@pytest.mark.parametrize("value, expected", [
    ("1", True), ("true", True), ("YES", True), (" on ", True),
    ("0", False), ("false", False), ("No", False), ("off", False), ("", False),
])
def test_parse_flag_known_spellings_map_to_booleans(value, expected):
    """Checks common true/false spellings are read case- and whitespace-insensitively."""
    assert parse_flag(value) is expected


def test_parse_flag_unknown_value_raises():
    """Checks a typo such as 'ture' fails instead of silently meaning False."""
    with pytest.raises(ValueError, match="boolean"):
        parse_flag("ture")


# ----- resolve_hub_options: precedence of argument, environment and default -----
def test_resolve_hub_options_empty_environment_uses_hf_default_online():
    """Checks no arguments and no variables give the Hugging Face default cache, online."""
    assert resolve_hub_options(environ={}) == HubOptions(cache_dir=None, local_files_only=False)


@pytest.mark.parametrize("cache_arg, offline_arg, environ, expected_cache, expected_offline", [
    # Environment only
    (None, None, {CACHE_ENV: "/env/cache"}, Path("/env/cache"), False),
    (None, None, {OFFLINE_ENV: "1"}, None, True),
    (None, None, {HF_OFFLINE_ENV: "1"}, None, True),
    # The project variable wins over the standard Hugging Face one
    (None, None, {OFFLINE_ENV: "0", HF_OFFLINE_ENV: "1"}, None, False),
    # Explicit arguments win over every variable
    ("/arg/cache", False, {CACHE_ENV: "/env/cache", OFFLINE_ENV: "1"}, Path("/arg/cache"), False),
    (None, True, {OFFLINE_ENV: "0"}, None, True),
])
def test_resolve_hub_options_precedence_is_argument_then_environment_then_default(
        cache_arg, offline_arg, environ, expected_cache, expected_offline):
    """Checks each setting comes from the argument, else the environment, else the default."""
    options = resolve_hub_options(cache_arg, offline_arg, environ=environ)

    assert options.cache_dir == expected_cache
    assert options.local_files_only is expected_offline


# ----- HubOptions.as_kwargs: what from_pretrained receives -----
@pytest.mark.parametrize("options, expected", [
    (HubOptions(), {"local_files_only": False}),
    (HubOptions(cache_dir=Path("cache"), local_files_only=True),
     {"local_files_only": True, "cache_dir": "cache"}),
])
def test_as_kwargs_includes_cache_dir_only_when_set(options, expected):
    """Checks cache_dir is omitted when unset, so Hugging Face keeps its own default and HF_HOME."""
    assert options.as_kwargs() == expected


# ----- export_hub_options: sharing settings with code that resolves later -----
@pytest.mark.parametrize("options", [
    HubOptions(cache_dir=Path("some/cache"), local_files_only=True),
    HubOptions(cache_dir=None, local_files_only=False),
])
def test_export_hub_options_round_trips_through_the_environment(options):
    """Checks exported options are exactly what a later resolve_hub_options call returns."""
    environ = {CACHE_ENV: "stale/cache", HF_OFFLINE_ENV: "1"}

    export_hub_options(options, environ)

    assert resolve_hub_options(environ=environ) == options


# ----- load_pinned_tokenizer: revision must be pinned -----
@pytest.mark.parametrize("revision", [None, ""])
def test_load_pinned_tokenizer_without_revision_raises(revision):
    """Checks an unpinned tokenizer is refused, since it could change chunk boundaries."""
    with pytest.raises(ValueError, match="pinned revision"):
        load_pinned_tokenizer("distilbert/distilroberta-base", revision)


def test_load_pinned_tokenizer_passes_revision_and_hub_options(monkeypatch):
    """Checks the name, pinned revision, fast flag, cache folder and offline flag reach from_pretrained."""
    calls = []
    monkeypatch.setattr(hub.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or "tokenizer")

    result = load_pinned_tokenizer("some/model", PINNED, HubOptions(cache_dir=Path("c"), local_files_only=True))

    assert result == "tokenizer"
    assert calls == [(("some/model",), {"revision": PINNED, "use_fast": True,
                                         "local_files_only": True, "cache_dir": "c"})]


def test_load_pinned_tokenizer_loads_real_tokenizer_offline():
    """Checks the pinned DistilRoBERTa tokenizer loads from the local cache with no network access."""
    tokenizer = load_pinned_tokenizer("distilbert/distilroberta-base", PINNED,
                                      HubOptions(local_files_only=True))

    assert tokenizer.is_fast
    assert tokenizer("Governing Law")["input_ids"][0] == tokenizer.cls_token_id
