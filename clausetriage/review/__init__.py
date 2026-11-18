"""
Human in the loop: the review queue, validated reviewer feedback and its export as training data.

Plain Python over an ArtifactStore with Pydantic models at the boundary (no HTTP or UI), so the API
and the reviewer UI can expose these functions directly. See docs/feedback-review-queue.md.

queue.py     build_review_queue(): Unknowns and reviewer conflicts, with chunk text, pages and status
feedback.py  submit_feedback(), resolve_feedback(), list_resolutions(): verdicts and how they resolve
export.py    build_training_manifest(), export_training_examples(), write_export(): leakage-safe export
schema.py    the Pydantic models all of them return
"""
