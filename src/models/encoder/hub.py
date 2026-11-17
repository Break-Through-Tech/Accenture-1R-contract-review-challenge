"""
Where the encoder path looks for Hugging Face files, and whether it may use the network.

Older scripts hard-coded cache_dir=<repo>/.venv/hf-cache with local_files_only=True, which
only works on the machine that filled that cache. The encoder path resolves both settings
instead, so the same code runs locally (offline, from a known cache) and on Colab (online,
default cache). The model revision is never part of these options: callers pass the pinned
revision explicitly, and load_pinned_tokenizer refuses to load without one.

Precedence for each setting: explicit argument, then environment variable, then default.
- cache directory: CLAUSETRIAGE_HF_CACHE, else None (the Hugging Face default, which honours HF_HOME)
- offline:         CLAUSETRIAGE_HF_OFFLINE, else HF_HUB_OFFLINE, else False
"""
from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase

CACHE_ENV = "CLAUSETRIAGE_HF_CACHE"
OFFLINE_ENV = "CLAUSETRIAGE_HF_OFFLINE"
HF_OFFLINE_ENV = "HF_HUB_OFFLINE"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


class HubOptions(BaseModel):
    """
    Cache location and offline flag for Hugging Face from_pretrained calls.
    cache_dir None means the Hugging Face default cache; local_files_only True means no
    network access, so a missing file fails immediately instead of downloading.
    """
    model_config = ConfigDict(frozen=True)

    cache_dir: Path | None = None
    local_files_only: bool = False

    def as_kwargs(self) -> dict:
        """Return the keyword arguments to pass to from_pretrained (cache_dir only when set)."""
        kwargs: dict = {"local_files_only": self.local_files_only}
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)
        return kwargs


def parse_flag(value: str) -> bool:
    """
    Parse an environment-variable boolean such as "1", "true" or "off" (case-insensitive).
    Raises ValueError on anything else, so a typo is not silently read as False.
    """
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Cannot read {value!r} as a boolean; use 1/0, true/false, yes/no or on/off")


def resolve_hub_options(
    cache_dir: str | Path | None = None,
    offline: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> HubOptions:
    """
    Resolve HubOptions from explicit arguments, then environment variables, then defaults.
    environ defaults to os.environ and can be passed explicitly in tests.
    """
    env = os.environ if environ is None else environ

    # Cache directory: argument, then CLAUSETRIAGE_HF_CACHE, then the Hugging Face default
    if cache_dir is None and env.get(CACHE_ENV):
        cache_dir = env[CACHE_ENV]

    # Offline flag: argument, then the project variable, then the standard Hugging Face one
    if offline is None:
        if env.get(OFFLINE_ENV) is not None:
            offline = parse_flag(env[OFFLINE_ENV])
        elif env.get(HF_OFFLINE_ENV) is not None:
            offline = parse_flag(env[HF_OFFLINE_ENV])
        else:
            offline = False

    return HubOptions(cache_dir=Path(cache_dir) if cache_dir is not None else None,
                      local_files_only=offline)


def export_hub_options(hub: HubOptions, environ: MutableMapping[str, str] | None = None) -> None:
    """
    Write hub options to the project environment variables so code that resolves its own
    options later in the same process (fit, build_model, load_data) uses the same settings.
    """
    env = os.environ if environ is None else environ
    if hub.cache_dir is None:
        env.pop(CACHE_ENV, None)
    else:
        env[CACHE_ENV] = str(hub.cache_dir)
    env[OFFLINE_ENV] = "1" if hub.local_files_only else "0"


def load_pinned_tokenizer(
    model_name: str,
    revision: str | None,
    hub: HubOptions | None = None,
) -> PreTrainedTokenizerBase:
    """
    Load the fast tokenizer for model_name at an exact revision, using the given hub options.
    Raises ValueError when revision is None, because an unpinned tokenizer can silently change
    chunk boundaries and break comparisons with saved results.
    """
    if not revision:
        raise ValueError(f"Refusing to load {model_name!r} without a pinned revision")
    hub = hub if hub is not None else resolve_hub_options()
    return AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True, **hub.as_kwargs())
