from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from storage import (
    get_db,
    Experiment,
    ExperimentParameter,
)

router = APIRouter(prefix="/experiments", tags=["experiments"])


class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    project: str = Field(..., min_length=1, max_length=255)
    parameters: Optional[Dict[str, Any]] = Field(default=None)
    description: Optional[str] = Field(default=None)


class ExperimentUpdate(BaseModel):
    status: Optional[str] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)


class ExperimentParameterResponse(BaseModel):
    key: str
    value: str


class ExperimentResponse(BaseModel):
    id: int
    name: str
    project: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    description: Optional[str]
    parameters: List[ExperimentParameterResponse] = []
    created_at: datetime

    class Config:
        from_attributes = True


def experiment_to_response(exp: Experiment) -> ExperimentResponse:
    params = [
        ExperimentParameterResponse(key=p.key, value=p.value)
        for p in exp.parameters
    ]
    return ExperimentResponse(
        id=exp.id,
        name=exp.name,
        project=exp.project,
        status=exp.status,
        start_time=exp.start_time,
        end_time=exp.end_time,
        description=exp.description,
        parameters=params,
        created_at=exp.created_at,
    )


@router.post("", response_model=ExperimentResponse)
def create_experiment(exp_create: ExperimentCreate, db: Session = Depends(get_db)):
    exp = Experiment(
        name=exp_create.name,
        project=exp_create.project,
        status="running",
        start_time=datetime.utcnow(),
        description=exp_create.description,
    )
    db.add(exp)
    db.flush()

    if exp_create.parameters:
        for key, value in exp_create.parameters.items():
            param = ExperimentParameter(
                experiment_id=exp.id,
                key=str(key),
                value=str(value),
            )
            db.add(param)

    db.commit()
    db.refresh(exp)
    return experiment_to_response(exp)


@router.get("", response_model=List[ExperimentResponse])
def list_experiments(
    project: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    query = db.query(Experiment)
    if project:
        query = query.filter(Experiment.project == project)
    if status:
        query = query.filter(Experiment.status == status)
    experiments = query.order_by(Experiment.created_at.desc()).offset(skip).limit(limit).all()
    return [experiment_to_response(exp) for exp in experiments]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment_to_response(exp)


@router.patch("/{experiment_id}", response_model=ExperimentResponse)
def update_experiment(
    experiment_id: int,
    exp_update: ExperimentUpdate,
    db: Session = Depends(get_db),
):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if exp_update.status:
        valid_statuses = {"running", "success", "failed"}
        if exp_update.status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {valid_statuses}",
            )
        exp.status = exp_update.status

    if exp_update.end_time:
        exp.end_time = exp_update.end_time
    elif exp_update.status in {"success", "failed"} and not exp.end_time:
        exp.end_time = datetime.utcnow()

    db.commit()
    db.refresh(exp)
    return experiment_to_response(exp)


@router.delete("/{experiment_id}")
def delete_experiment(experiment_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    db.delete(exp)
    db.commit()
    return {"message": "Experiment deleted successfully"}
