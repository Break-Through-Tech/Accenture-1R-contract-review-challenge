"""Weight endpoints: list, read and create immutable weight versions, and preview a re-ranking."""
from fastapi import APIRouter, Response

from src.api.routers import ServiceDep
from src.api.schemas import ContractRanking, PreviewRequest, WeightVersionCreated, WeightVersionDetail, WeightVersionList
from src.risk.schema import WeightSet

router = APIRouter(tags=["weights"])


@router.get("/weights", response_model=WeightVersionList)
def list_weights(service: ServiceDep) -> WeightVersionList:
    """Every stored weight version (oldest first) and the default one new uploads are scored with."""
    return service.list_weights()


@router.post("/weights", status_code=201, response_model=WeightVersionCreated)
def create_weights(weight_set: WeightSet, response: Response, service: ServiceDep) -> WeightVersionCreated:
    """
    Create a new weight version and score every stored contract with it from cached probabilities.
    Never changes an existing version: the same id with other values is a 409; identical content is a 200.
    """
    created = service.create_weights(weight_set)
    if not created.created:
        response.status_code = 200
    return created


@router.post("/weights/preview", response_model=ContractRanking)
def preview_weights(request: PreviewRequest, service: ServiceDep) -> ContractRanking:
    """Rank stored contracts under a candidate weight set without saving anything."""
    return service.preview(request.weight_set, request.contract_ids, request.model_version)


@router.get("/weights/{weight_version_id}", response_model=WeightVersionDetail)
def get_weights(weight_version_id: str, service: ServiceDep) -> WeightVersionDetail:
    """One weight version: created_by, created_at, content hash and the full weight set."""
    return service.get_weights(weight_version_id)
