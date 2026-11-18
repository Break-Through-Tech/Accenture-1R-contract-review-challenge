"""
FastAPI routers, one module per resource. Each handler parses the request, calls one
ClauseTriageService method (through the get_service dependency) and returns its model.

A new router needs no change here or in the existing handlers: define an APIRouter that depends
on get_service and pass it to create_app(service, routers=[...]).
"""
from typing import Annotated

from fastapi import Depends, Request

from clausetriage.api.service import ClauseTriageService


def get_service(request: Request) -> ClauseTriageService:
    """The ClauseTriageService create_app() attached to the application."""
    return request.app.state.service


ServiceDep = Annotated[ClauseTriageService, Depends(get_service)]
