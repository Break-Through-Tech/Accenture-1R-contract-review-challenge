"""Server endpoints: an index of links, health, and the model versions with stored results."""
from fastapi import APIRouter

from src.api.routers import ServiceDep
from src.api.schemas import HealthResponse, ModelVersionList

router = APIRouter(tags=["server"])


@router.get("/")
def index() -> dict[str, str]:
    """Where to find the API documentation and the reviewer page."""
    return {"name": "ClauseTriage API", "docs": "/docs", "openapi": "/openapi.json", "ui": "/ui/",
            "note": "Local research prototype without authentication: do not expose it on a network."}


@router.get("/health", response_model=HealthResponse)
def health(service: ServiceDep) -> HealthResponse:
    """Liveness, the server's model version and the default weight version."""
    return HealthResponse(model_version=service.model_version,
                          default_weight_version_id=service.default_weight_set.weight_version_id)


@router.get("/models", response_model=ModelVersionList)
def list_models(service: ServiceDep) -> ModelVersionList:
    """Every model version with stored results, and the one this server scores uploads with."""
    return service.list_model_versions()
