"""Run the pinned PR #48 prototype against our four strategies on the saved development IDs."""

from scripts.evaluate_token_chunking import main


if __name__ == "__main__":
    main(include_gwen=True)
