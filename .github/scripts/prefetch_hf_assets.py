"""
Download the Hugging Face files the test suite reads, so pytest can then run with HF_HUB_OFFLINE=1.

The tests load only the distilbert/distilroberta-base tokenizer (no model weights), from two caches:
- `.venv/hf-cache` under the repo root, at the pinned TOKENIZER_REVISION (the README's setup cache);
- the default Hugging Face cache (HF_HOME), at whatever revision `main` points to.
Files already cached are reused without a network call, so a warm CI cache never contacts the Hub.

Usage, from the repo root: python .github/scripts/prefetch_hf_assets.py
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Import the pinned name and revision from the code instead of repeating them here
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from clausetriage.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION  # noqa: E402

# Everything AutoTokenizer reads for this model; the weights are deliberately left out
TOKENIZER_FILES = ["config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"]
PINNED_CACHE = ROOT / ".venv" / "hf-cache"


def prefetch(revision: str, cache_dir: Path | None) -> str:
    """Return the local snapshot folder for `revision`, downloading the tokenizer files only if missing."""
    # Try the cache first so a warm cache needs no network; a partial snapshot counts as missing
    try:
        folder = snapshot_download(TOKENIZER_NAME, revision=revision, cache_dir=cache_dir,
                                   allow_patterns=TOKENIZER_FILES, local_files_only=True)
        if all((Path(folder) / name).is_file() for name in TOKENIZER_FILES):
            return folder
    except Exception:  # nothing cached yet (the exact error type varies by huggingface_hub version)
        pass
    # One file at a time: parallel workers race on Windows when symlinks are unavailable
    return snapshot_download(TOKENIZER_NAME, revision=revision, cache_dir=cache_dir,
                             allow_patterns=TOKENIZER_FILES, max_workers=1)


def main() -> None:
    """Fill both caches the tests read and print where each snapshot is."""
    print(f"pinned {TOKENIZER_REVISION[:12]}: {prefetch(TOKENIZER_REVISION, PINNED_CACHE)}")
    print(f"main (default cache): {prefetch('main', None)}")


if __name__ == "__main__":
    main()
