import pickle
import io
from datetime import datetime
from typing import Optional, List, Any
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from storage import (
    get_db,
    Experiment,
    ModelVersion,
    FileStorage,
    get_next_version,
    clear_production_flag,
)

router = APIRouter(prefix="/models", tags=["models"])


class ModelVersionResponse(BaseModel):
    id: int
    experiment_id: int
    experiment_name: str
    name: str
    version: int
    file_path: str
    file_size: int
    checksum: str
    is_production: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ProductionSetResponse(BaseModel):
    message: str
    model_id: int
    name: str
    version: int


def _to_response(mv: ModelVersion) -> ModelVersionResponse:
    return ModelVersionResponse(
        id=mv.id,
        experiment_id=mv.experiment_id,
        experiment_name=mv.experiment.name if mv.experiment else "",
        name=mv.name,
        version=mv.version,
        file_path=mv.file_path,
        file_size=mv.file_size,
        checksum=mv.checksum,
        is_production=mv.is_production,
        created_at=mv.created_at,
    )


@router.post("/register/{experiment_id}", response_model=ModelVersionResponse)
async def register_model(
    experiment_id: int,
    name: str = Form(..., min_length=1, max_length=255),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    version = get_next_version(db, name)
    content = await file.read()

    file_path, file_size, checksum = FileStorage.save_model_file(
        content, experiment_id, name, version
    )

    mv = ModelVersion(
        experiment_id=experiment_id,
        name=name,
        version=version,
        file_path=file_path,
        file_size=file_size,
        checksum=checksum,
        is_production=False,
        created_at=datetime.utcnow(),
    )
    db.add(mv)
    db.commit()
    db.refresh(mv)
    return _to_response(mv)


@router.post("/register-sklearn/{experiment_id}", response_model=ModelVersionResponse)
async def register_sklearn_model(
    experiment_id: int,
    name: str = Form(..., min_length=1, max_length=255),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    content = await file.read()
    try:
        model = pickle.loads(content)
        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise ValueError("Not a valid scikit-learn model")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scikit-learn model file: {str(e)}",
        )

    version = get_next_version(db, name)
    file_path, file_size, checksum = FileStorage.save_model_file(
        content, experiment_id, name, version
    )

    mv = ModelVersion(
        experiment_id=experiment_id,
        name=name,
        version=version,
        file_path=file_path,
        file_size=file_size,
        checksum=checksum,
        is_production=False,
        created_at=datetime.utcnow(),
    )
    db.add(mv)
    db.commit()
    db.refresh(mv)
    return _to_response(mv)


@router.get("", response_model=List[ModelVersionResponse])
def list_models(
    name: Optional[str] = Query(default=None),
    experiment_id: Optional[int] = Query(default=None),
    only_production: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    query = db.query(ModelVersion)
    if name:
        query = query.filter(ModelVersion.name == name)
    if experiment_id:
        query = query.filter(ModelVersion.experiment_id == experiment_id)
    if only_production:
        query = query.filter(ModelVersion.is_production == True)
    models = query.order_by(ModelVersion.name, ModelVersion.version.desc()).all()
    return [_to_response(m) for m in models]


@router.get("/names", response_model=List[str])
def list_model_names(db: Session = Depends(get_db)):
    names = db.query(ModelVersion.name).distinct().all()
    return [n[0] for n in names]


@router.get("/{model_id}", response_model=ModelVersionResponse)
def get_model(model_id: int, db: Session = Depends(get_db)):
    mv = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")
    return _to_response(mv)


@router.get("/{model_id}/download")
def download_model(model_id: int, db: Session = Depends(get_db)):
    mv = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")

    content = FileStorage.get_model_file(mv.file_path)
    if not content:
        raise HTTPException(status_code=404, detail="Model file not found on disk")

    filename = f"{mv.name}_v{mv.version}.pkl"
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/{model_id}/production", response_model=ProductionSetResponse)
def set_production(model_id: int, db: Session = Depends(get_db)):
    mv = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")

    clear_production_flag(db, mv.name)
    mv.is_production = True
    db.commit()
    db.refresh(mv)

    return ProductionSetResponse(
        message=f"Model {mv.name} v{mv.version} set as production",
        model_id=mv.id,
        name=mv.name,
        version=mv.version,
    )


@router.get("/production/{model_name}", response_model=ModelVersionResponse)
def get_production_model(model_name: str, db: Session = Depends(get_db)):
    mv = (
        db.query(ModelVersion)
        .filter(ModelVersion.name == model_name, ModelVersion.is_production == True)
        .first()
    )
    if not mv:
        raise HTTPException(
            status_code=404,
            detail=f"No production model found for name: {model_name}",
        )
    return _to_response(mv)


@router.delete("/{model_id}")
def delete_model(model_id: int, db: Session = Depends(get_db)):
    mv = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")

    FileStorage.delete_model_file(mv.file_path)
    db.delete(mv)
    db.commit()
    return {"message": "Model version deleted successfully"}
