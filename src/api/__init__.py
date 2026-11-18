"""
HTTP serving layer for ClauseTriage: the one place clients read pipeline outputs from.

How everything comes together:
service.py:  ClauseTriageService, the plain-Python layer every endpoint calls (scoring jobs, ranked
             reads, contract detail, registers, weight versions, preview); usable in-process
jobs.py:     JobStore, which records each upload's scoring status next to the artifact store
schemas.py:  the Pydantic request and response models
limits.py:   BodySizeLimitMiddleware, which refuses oversized request bodies before they are read
routers/:    thin FastAPI handlers over the service, one module per resource
app.py:      create_app(), which wires the service, routers, error handlers and the Gradio page
runtime.py:  loads the model, tokenizer and weights and builds a service from command-line settings
__main__.py: python -m src.api, the local server (bound to 127.0.0.1 by default)

No authentication: a local research prototype that must not be exposed on a network (docs/api.md).
"""
