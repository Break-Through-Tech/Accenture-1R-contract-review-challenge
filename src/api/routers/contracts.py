"""Contract endpoints: upload or paste a contract, poll its scoring job, read rankings, details and registers."""
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Query, Response, UploadFile

from src.api.jobs import JobRecord
from src.api.routers import ServiceDep
from src.api.schemas import ContractDetailResponse, ContractRanking, PastedTextRequest
from src.api.service import TooLargeError
from src.ingest.documents import MAX_FILE_BYTES, MB
from src.register.export import register_file_stem, register_frame, to_json

router = APIRouter(tags=["contracts"])

ModelVersion = Annotated[str | None, Query(description="Model version (default: the server's)")]
WeightVersion = Annotated[str | None, Query(description="Weight version (default: the default weight version)")]


def _queued(job: JobRecord, queued: bool, background: BackgroundTasks, response: Response,
            service: ServiceDep) -> JobRecord:
    """Schedule a newly queued job after the response (202); an existing job is returned as is (200)."""
    if queued:
        background.add_task(service.run_job, job.contract_id)
    else:
        response.status_code = 200
    return job


# ----- Submitting contracts -----
@router.post("/contracts/upload", status_code=202, response_model=JobRecord)
def upload_contract(file: Annotated[UploadFile, File(description="A .pdf, .docx or .txt contract")],
                    background: BackgroundTasks, response: Response, service: ServiceDep) -> JobRecord:
    """Upload a PDF, Word or text contract; it is read now and scored in the background. Poll its status."""
    if file.size is not None and file.size > MAX_FILE_BYTES:
        raise TooLargeError(f"This file is over the {MAX_FILE_BYTES // MB} MB limit.")
    data = file.file.read(MAX_FILE_BYTES + 1)
    job, queued = service.submit_upload(data, file.filename)
    return _queued(job, queued, background, response, service)


@router.post("/contracts/text", status_code=202, response_model=JobRecord)
def paste_contract(request: PastedTextRequest, background: BackgroundTasks, response: Response,
                   service: ServiceDep) -> JobRecord:
    """Submit pasted contract text, scored exactly as given in the background. Poll its status."""
    job, queued = service.submit_text(request.text, request.name)
    return _queued(job, queued, background, response, service)


@router.get("/jobs", response_model=list[JobRecord])
def list_jobs(service: ServiceDep) -> list[JobRecord]:
    """Every scoring job under the server's model version, most recent first."""
    return service.list_jobs()


@router.get("/contracts/{contract_id}/status", response_model=JobRecord)
def contract_status(contract_id: str, service: ServiceDep) -> JobRecord:
    """A contract's scoring status: queued, running, complete or failed (with the reason)."""
    return service.get_job(contract_id)


# ----- Reading -----
@router.get("/contracts", response_model=ContractRanking)
def list_contracts(service: ServiceDep, model_version: ModelVersion = None,
                   weight_version_id: WeightVersion = None) -> ContractRanking:
    """Every scored contract ranked for review (rank 1 = open first)."""
    return service.list_ranked(model_version, weight_version_id)


@router.get("/contracts/{contract_id}/register.csv", response_class=Response)
def register_csv(contract_id: str, service: ServiceDep, model_version: ModelVersion = None,
                 weight_version_id: WeightVersion = None) -> Response:
    """The contract's clause register as CSV (the columns of docs/clause-register.md)."""
    register = service.get_register(contract_id, model_version, weight_version_id)
    body = register_frame([register]).to_csv(index=False, lineterminator="\n")
    return _download(body, "text/csv; charset=utf-8", f"{register_file_stem(contract_id)}.csv")


@router.get("/contracts/{contract_id}/register.json", response_class=Response)
def register_json(contract_id: str, service: ServiceDep, model_version: ModelVersion = None,
                  weight_version_id: WeightVersion = None) -> Response:
    """The contract's clause register as JSON (header and rows; read back with register.export.read_json)."""
    register = service.get_register(contract_id, model_version, weight_version_id)
    return _download(to_json(register), "application/json", f"{register_file_stem(contract_id)}.json")


@router.get("/contracts/{contract_id}", response_model=ContractDetailResponse)
def contract_detail(contract_id: str, service: ServiceDep, model_version: ModelVersion = None,
                    weight_version_id: WeightVersion = None) -> ContractDetailResponse:
    """One contract: text, chunks with probabilities, detections, evidence, Unknowns, scores and rank."""
    return service.get_contract_detail(contract_id, model_version, weight_version_id)


def _download(body: str, media_type: str, filename: str) -> Response:
    """A UTF-8 file download; filename is a register_file_stem, so it is always ASCII and header-safe."""
    return Response(content=body.encode("utf-8"), media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})
