import pickle
import io
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Dict, Union, Literal
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from sqlalchemy.exc import IntegrityError

from storage import (
    get_db,
    Experiment,
    ModelVersion,
    FileStorage,
    get_next_version,
    clear_production_flag,
)

router = APIRouter(prefix="/models", tags=["models"])


_model_cache_lock = threading.Lock()
_model_cache: Dict[int, Dict[str, Any]] = {}


def _get_file_mtime(file_path: str) -> Optional[float]:
    p = Path(file_path)
    return p.stat().st_mtime if p.exists() else None


def _load_model_instance(file_path: str, checksum: str) -> Any:
    content = FileStorage.get_model_file(file_path)
    if not content:
        raise HTTPException(status_code=404, detail="Model file not found on disk")
    try:
        model = pickle.loads(content)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unpickle model: {str(e)}",
        )
    if not hasattr(model, "predict"):
        raise HTTPException(
            status_code=400,
            detail="Model does not have a predict() method",
        )
    return model


def _detect_prediction_type(model: Any) -> Literal["classification", "regression"]:
    if hasattr(model, "predict_proba"):
        return "classification"
    return "regression"


def get_cached_model(model_id: int, file_path: str, checksum: str) -> Any:
    with _model_cache_lock:
        entry = _model_cache.get(model_id)
        current_mtime = _get_file_mtime(file_path)
        if (
            entry is not None
            and entry.get("checksum") == checksum
            and entry.get("mtime") == current_mtime
        ):
            return entry["model"]

        model = _load_model_instance(file_path, checksum)
        pred_type = _detect_prediction_type(model)
        classes = list(model.classes_) if hasattr(model, "classes_") else None
        _model_cache[model_id] = {
            "model": model,
            "checksum": checksum,
            "mtime": current_mtime,
            "prediction_type": pred_type,
            "classes": classes,
        }
        return model


def get_cached_model_info(model_id: int) -> Optional[Dict[str, Any]]:
    with _model_cache_lock:
        return _model_cache.get(model_id)


def get_cached_model_classes(model_id: int) -> Optional[List[Any]]:
    with _model_cache_lock:
        entry = _model_cache.get(model_id)
        return entry.get("classes") if entry else None


def invalidate_model_cache(model_id: Optional[int] = None, model_name: Optional[str] = None, db: Optional[Session] = None):
    with _model_cache_lock:
        if model_id is not None:
            _model_cache.pop(model_id, None)
            return
        if model_name is not None and db is not None:
            ids_to_remove = [
                m.id for m in db.query(ModelVersion).filter(ModelVersion.name == model_name).all()
            ]
            for mid in ids_to_remove:
                _model_cache.pop(mid, None)


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


class PredictRequest(BaseModel):
    features: List[List[float]] = Field(..., description="二维数组，每行一个样本")
    feature_names: Optional[List[str]] = Field(default=None)
    predict_proba: bool = Field(default=False, description="分类模型时是否返回概率")


class PredictResponse(BaseModel):
    model_id: int
    model_name: str
    version: int
    prediction_type: Literal["classification", "regression"]
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    classes: Optional[List[Any]] = None


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


def _resolve_model(
    db: Session,
    model_id: Optional[int] = None,
    model_name: Optional[str] = None,
    production: bool = False,
) -> ModelVersion:
    if model_id is not None:
        mv = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
        if not mv:
            raise HTTPException(status_code=404, detail="Model version not found")
        return mv
    if model_name:
        if production:
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
            return mv
        mv = (
            db.query(ModelVersion)
            .filter(ModelVersion.name == model_name)
            .order_by(ModelVersion.version.desc())
            .first()
        )
        if not mv:
            raise HTTPException(status_code=404, detail=f"No model found for name: {model_name}")
        return mv
    raise HTTPException(status_code=400, detail="Either model_id or model_name must be provided")


def _run_prediction(mv: ModelVersion, features: List[List[float]], predict_proba: bool) -> PredictResponse:
    model = get_cached_model(mv.id, mv.file_path, mv.checksum)
    info = get_cached_model_info(mv.id) or {}
    pred_type = info.get("prediction_type") or _detect_prediction_type(model)
    classes = get_cached_model_classes(mv.id)

    X = np.array(features, dtype=float)
    preds = model.predict(X)
    preds_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)

    if pred_type == "classification" and classes is not None:
        preds_list = [
            classes[int(p)] if isinstance(p, (int, float, np.integer, np.floating)) else p
            for p in preds_list
        ]

    result = PredictResponse(
        model_id=mv.id,
        model_name=mv.name,
        version=mv.version,
        prediction_type=pred_type,
        predictions=preds_list,
    )

    if predict_proba and pred_type == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        result.probabilities = proba.tolist() if hasattr(proba, "tolist") else [list(p) for p in proba]
        result.classes = classes

    return result


MAX_RETRIES = 3


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

    content = await file.read()

    last_error = None
    for attempt in range(MAX_RETRIES):
        version = get_next_version(db, name)
        file_path, file_size, checksum = FileStorage.save_model_file(
            content, experiment_id, name, version
        )
        try:
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
            invalidate_model_cache(model_name=name, db=db)
            return _to_response(mv)
        except IntegrityError as e:
            db.rollback()
            FileStorage.delete_file(file_path)
            last_error = e
            if attempt < MAX_RETRIES - 1:
                continue
            raise HTTPException(
                status_code=500,
                detail=f"Failed to register model after {MAX_RETRIES} attempts due to version conflict: {str(last_error)}",
            )


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

    last_error = None
    for attempt in range(MAX_RETRIES):
        version = get_next_version(db, name)
        file_path, file_size, checksum = FileStorage.save_model_file(
            content, experiment_id, name, version
        )
        try:
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
            invalidate_model_cache(model_name=name, db=db)
            return _to_response(mv)
        except IntegrityError as e:
            db.rollback()
            FileStorage.delete_file(file_path)
            last_error = e
            if attempt < MAX_RETRIES - 1:
                continue
            raise HTTPException(
                status_code=500,
                detail=f"Failed to register model after {MAX_RETRIES} attempts due to version conflict: {str(last_error)}",
            )


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

    invalidate_model_cache(model_name=mv.name, db=db)

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
    name = mv.name
    db.delete(mv)
    db.commit()
    invalidate_model_cache(model_id=model_id)
    invalidate_model_cache(model_name=name, db=db)
    return {"message": "Model version deleted successfully"}


@router.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    model_id: Optional[int] = Query(default=None),
    model_name: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    mv = _resolve_model(db, model_id=model_id, model_name=model_name, production=True)
    if not mv.is_production:
        raise HTTPException(
            status_code=400,
            detail="Only production models can be used for online prediction. Set this model as production first.",
        )
    if not req.features:
        raise HTTPException(status_code=400, detail="features must not be empty")
    return _run_prediction(mv, req.features, req.predict_proba)


@router.post("/predict/{model_id}", response_model=PredictResponse)
def predict_by_model_id(
    model_id: int,
    req: PredictRequest,
    db: Session = Depends(get_db),
):
    mv = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")
    if not req.features:
        raise HTTPException(status_code=400, detail="features must not be empty")
    return _run_prediction(mv, req.features, req.predict_proba)


@router.get("/cache/stats")
def cache_stats():
    with _model_cache_lock:
        return {
            "cached_models": list(_model_cache.keys()),
            "count": len(_model_cache),
        }


@router.post("/cache/clear")
def cache_clear(
    model_id: Optional[int] = Query(default=None),
    model_name: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    invalidate_model_cache(model_id=model_id, model_name=model_name, db=db)
    return {"message": "Cache cleared"}
