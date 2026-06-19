from datetime import datetime
from typing import Optional, Dict, Any, List, Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func

from storage import (
    get_db,
    Experiment,
    ExperimentParameter,
    ExperimentTag,
    Metric,
)

router = APIRouter(prefix="/experiments", tags=["experiments"])


class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    project: str = Field(..., min_length=1, max_length=255)
    parameters: Optional[Dict[str, Any]] = Field(default=None)
    description: Optional[str] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None)


class ExperimentUpdate(BaseModel):
    status: Optional[str] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)
    description: Optional[str] = Field(default=None)


class ExperimentParameterResponse(BaseModel):
    key: str
    value: str


class TagResponse(BaseModel):
    id: int
    tag: str
    created_at: datetime


class ExperimentResponse(BaseModel):
    id: int
    name: str
    project: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    description: Optional[str]
    parameters: List[ExperimentParameterResponse] = []
    tags: List[str] = []
    created_at: datetime

    class Config:
        from_attributes = True


class TagsAddRemove(BaseModel):
    tags: List[str] = Field(..., min_length=1)


def experiment_to_response(exp: Experiment) -> ExperimentResponse:
    params = [
        ExperimentParameterResponse(key=p.key, value=p.value)
        for p in exp.parameters
    ]
    tags = sorted([t.tag for t in exp.tags])
    return ExperimentResponse(
        id=exp.id,
        name=exp.name,
        project=exp.project,
        status=exp.status,
        start_time=exp.start_time,
        end_time=exp.end_time,
        description=exp.description,
        parameters=params,
        tags=tags,
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

    if exp_create.tags:
        for tag_name in exp_create.tags:
            tag = ExperimentTag(experiment_id=exp.id, tag=str(tag_name))
            db.add(tag)

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


@router.get("/search", response_model=List[ExperimentResponse])
def search_experiments(
    project: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    tags: Optional[List[str]] = Query(default=None),
    tag_mode: Literal["any", "all"] = Query(default="any", description="any: 匹配任一标签；all: 必须匹配全部标签"),
    name_contains: Optional[str] = Query(default=None),
    created_from: Optional[datetime] = Query(default=None),
    created_to: Optional[datetime] = Query(default=None),
    sort_by_metric: Optional[str] = Query(default=None, description="按某个指标的最新值排序"),
    sort_order: Literal["desc", "asc"] = Query(default="desc"),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    query = db.query(Experiment)

    if project:
        query = query.filter(Experiment.project == project)
    if status:
        query = query.filter(Experiment.status == status)
    if name_contains:
        query = query.filter(Experiment.name.contains(name_contains))
    if created_from:
        query = query.filter(Experiment.created_at >= created_from)
    if created_to:
        query = query.filter(Experiment.created_at <= created_to)
    if tags:
        unique_tags = list(set(tags))
        if tag_mode == "all":
            for t in unique_tags:
                subq = (
                    db.query(ExperimentTag.experiment_id)
                    .filter(ExperimentTag.tag == t)
                    .subquery()
                )
                query = query.filter(Experiment.id.in_(subq))
        else:
            subq = (
                db.query(ExperimentTag.experiment_id)
                .filter(ExperimentTag.tag.in_(unique_tags))
                .subquery()
            )
            query = query.filter(Experiment.id.in_(subq))

    if sort_by_metric:
        metric_subq = (
            db.query(
                Metric.experiment_id,
                func.max(Metric.step).label("max_step"),
            )
            .filter(Metric.name == sort_by_metric)
            .group_by(Metric.experiment_id)
            .subquery()
        )
        value_subq = (
            db.query(
                Metric.experiment_id,
                Metric.value.label("metric_value"),
            )
            .join(
                metric_subq,
                and_(
                    Metric.experiment_id == metric_subq.c.experiment_id,
                    Metric.step == metric_subq.c.max_step,
                ),
            )
            .filter(Metric.name == sort_by_metric)
            .subquery()
        )
        order_func = desc if sort_order == "desc" else asc
        query = query.outerjoin(
            value_subq, Experiment.id == value_subq.c.experiment_id
        ).order_by(
            value_subq.c.metric_value.is_(None),
            order_func(value_subq.c.metric_value),
            Experiment.created_at.desc(),
        )
    else:
        query = query.order_by(Experiment.created_at.desc())

    experiments = query.offset(skip).limit(limit).all()
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

    if exp_update.description is not None:
        exp.description = exp_update.description

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


@router.get("/{experiment_id}/tags", response_model=List[str])
def list_experiment_tags(experiment_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return sorted([t.tag for t in exp.tags])


@router.post("/{experiment_id}/tags", response_model=List[str])
def add_experiment_tags(
    experiment_id: int,
    body: TagsAddRemove,
    db: Session = Depends(get_db),
):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    existing = {t.tag for t in exp.tags}
    for tag_name in body.tags:
        if tag_name not in existing:
            tag = ExperimentTag(experiment_id=experiment_id, tag=str(tag_name))
            db.add(tag)
    db.commit()
    db.refresh(exp)
    return sorted([t.tag for t in exp.tags])


@router.delete("/{experiment_id}/tags", response_model=List[str])
def remove_experiment_tags(
    experiment_id: int,
    body: TagsAddRemove,
    db: Session = Depends(get_db),
):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    tags_to_remove = set(body.tags)
    for t in list(exp.tags):
        if t.tag in tags_to_remove:
            db.delete(t)
    db.commit()
    db.refresh(exp)
    return sorted([t.tag for t in exp.tags])


@router.get("/tags/all", response_model=List[str])
def list_all_tags(
    project: Optional[str] = Query(default=None), db: Session = Depends(get_db)
):
    query = db.query(ExperimentTag.tag)
    if project:
        query = query.join(Experiment, ExperimentTag.experiment_id == Experiment.id).filter(
            Experiment.project == project
        )
    tags = query.distinct().order_by(ExperimentTag.tag).all()
    return [t[0] for t in tags]
