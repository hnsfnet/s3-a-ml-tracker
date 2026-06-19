from datetime import datetime
from typing import Optional, List, Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func

from storage import get_db, Experiment, Metric

router = APIRouter(prefix="/metrics", tags=["metrics"])


class MetricCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    value: float
    step: Optional[int] = Field(default=0)


class MetricResponse(BaseModel):
    id: int
    experiment_id: int
    name: str
    value: float
    step: int
    timestamp: datetime

    class Config:
        from_attributes = True


class MetricRankingResponse(BaseModel):
    experiment_id: int
    experiment_name: str
    project: str
    metric_name: str
    value: float
    step: int
    timestamp: datetime


class LatestMetricResponse(BaseModel):
    experiment_id: int
    experiment_name: str
    project: str
    metric_name: str
    latest_value: float
    latest_step: int
    latest_timestamp: datetime


def _check_experiment(db: Session, experiment_id: int):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@router.post("/{experiment_id}", response_model=MetricResponse)
def log_metric(
    experiment_id: int,
    metric_create: MetricCreate,
    db: Session = Depends(get_db),
):
    _check_experiment(db, experiment_id)
    metric = Metric(
        experiment_id=experiment_id,
        name=metric_create.name,
        value=metric_create.value,
        step=metric_create.step or 0,
        timestamp=datetime.utcnow(),
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


@router.post("/{experiment_id}/batch", response_model=List[MetricResponse])
def log_metrics_batch(
    experiment_id: int,
    metrics_batch: List[MetricCreate],
    db: Session = Depends(get_db),
):
    _check_experiment(db, experiment_id)
    now = datetime.utcnow()
    created = []
    for item in metrics_batch:
        metric = Metric(
            experiment_id=experiment_id,
            name=item.name,
            value=item.value,
            step=item.step or 0,
            timestamp=now,
        )
        db.add(metric)
        created.append(metric)
    db.commit()
    for m in created:
        db.refresh(m)
    return created


@router.get("/{experiment_id}", response_model=List[MetricResponse])
def get_experiment_metrics(
    experiment_id: int,
    metric_name: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    _check_experiment(db, experiment_id)
    query = db.query(Metric).filter(Metric.experiment_id == experiment_id)
    if metric_name:
        query = query.filter(Metric.name == metric_name)
    metrics = query.order_by(Metric.name, Metric.step, Metric.timestamp).all()
    return metrics


@router.get("/{experiment_id}/names", response_model=List[str])
def get_metric_names(
    experiment_id: int,
    db: Session = Depends(get_db),
):
    _check_experiment(db, experiment_id)
    names = (
        db.query(Metric.name)
        .filter(Metric.experiment_id == experiment_id)
        .distinct()
        .all()
    )
    return [n[0] for n in names]


@router.get("/ranking/{metric_name}", response_model=List[LatestMetricResponse])
def get_metric_ranking(
    metric_name: str,
    project: Optional[str] = Query(default=None),
    order: Literal["desc", "asc"] = Query(default="desc"),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    order_func = desc if order == "desc" else asc

    subquery = (
        db.query(
            Metric.experiment_id,
            Metric.name,
            func.max(Metric.step).label("max_step"),
        )
        .filter(Metric.name == metric_name)
        .group_by(Metric.experiment_id, Metric.name)
        .subquery()
    )

    query = (
        db.query(
            Metric.experiment_id,
            Experiment.name.label("experiment_name"),
            Experiment.project,
            Metric.name.label("metric_name"),
            Metric.value.label("latest_value"),
            Metric.step.label("latest_step"),
            Metric.timestamp.label("latest_timestamp"),
        )
        .join(Experiment, Metric.experiment_id == Experiment.id)
        .join(
            subquery,
            (Metric.experiment_id == subquery.c.experiment_id)
            & (Metric.name == subquery.c.name)
            & (Metric.step == subquery.c.max_step),
        )
        .filter(Metric.name == metric_name)
    )

    if project:
        query = query.filter(Experiment.project == project)

    query = query.order_by(order_func(Metric.value)).limit(limit)
    results = query.all()

    return [
        LatestMetricResponse(
            experiment_id=r.experiment_id,
            experiment_name=r.experiment_name,
            project=r.project,
            metric_name=r.metric_name,
            latest_value=r.latest_value,
            latest_step=r.latest_step,
            latest_timestamp=r.latest_timestamp,
        )
        for r in results
    ]
