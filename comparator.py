from typing import Optional, List, Dict, Any, Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from storage import get_db, Experiment, ExperimentParameter, Metric

SQLITE_BATCH_SIZE = 500


def _batch_ids(ids: List[int], batch_size: int = SQLITE_BATCH_SIZE) -> List[List[int]]:
    return [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

router = APIRouter(prefix="/comparator", tags=["comparator"])


class ExperimentInfo(BaseModel):
    id: int
    name: str
    project: str
    status: str
    start_time: Any
    end_time: Optional[Any]


class ParameterDiff(BaseModel):
    key: str
    values: Dict[int, Optional[str]]


class MetricSummary(BaseModel):
    metric_name: str
    values: Dict[int, Optional[float]]
    best_experiment_id: Optional[int]
    best_value: Optional[float]


class ComparisonResult(BaseModel):
    experiments: List[ExperimentInfo]
    parameter_diffs: List[ParameterDiff]
    metric_summaries: List[MetricSummary]
    sorted_metrics: List[MetricSummary]


def _get_experiments(db: Session, experiment_ids: List[int]) -> List[Experiment]:
    experiments: List[Experiment] = []
    for batch in _batch_ids(experiment_ids):
        batch_exps = db.query(Experiment).filter(Experiment.id.in_(batch)).all()
        experiments.extend(batch_exps)
    found_ids = {exp.id for exp in experiments}
    missing = set(experiment_ids) - found_ids
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Experiments not found: {sorted(missing)}",
        )
    return experiments


def _collect_parameters(
    db: Session, experiment_ids: List[int]
) -> Dict[int, Dict[str, str]]:
    result: Dict[int, Dict[str, str]] = {eid: {} for eid in experiment_ids}
    for batch in _batch_ids(experiment_ids):
        params = (
            db.query(ExperimentParameter)
            .filter(ExperimentParameter.experiment_id.in_(batch))
            .all()
        )
        for p in params:
            result[p.experiment_id][p.key] = p.value
    return result


def _collect_latest_metrics(
    db: Session, experiment_ids: List[int]
) -> Dict[int, Dict[str, float]]:
    result: Dict[int, Dict[str, float]] = {eid: {} for eid in experiment_ids}
    for batch in _batch_ids(experiment_ids):
        subquery = (
            db.query(
                Metric.experiment_id,
                Metric.name,
                func.max(Metric.step).label("max_step"),
            )
            .filter(Metric.experiment_id.in_(batch))
            .group_by(Metric.experiment_id, Metric.name)
            .subquery()
        )

        metrics = (
            db.query(Metric)
            .join(
                subquery,
                (Metric.experiment_id == subquery.c.experiment_id)
                & (Metric.name == subquery.c.name)
                & (Metric.step == subquery.c.max_step),
            )
            .filter(Metric.experiment_id.in_(batch))
            .all()
        )
        for m in metrics:
            result[m.experiment_id][m.name] = m.value
    return result


def _build_parameter_diffs(
    experiment_ids: List[int],
    params_by_exp: Dict[int, Dict[str, str]],
) -> List[ParameterDiff]:
    all_keys = set()
    for exp_params in params_by_exp.values():
        all_keys.update(exp_params.keys())

    diffs: List[ParameterDiff] = []
    for key in sorted(all_keys):
        values: Dict[int, Optional[str]] = {}
        for eid in experiment_ids:
            values[eid] = params_by_exp[eid].get(key, None)
        unique_values = set(v for v in values.values() if v is not None)
        if len(unique_values) > 1 or len(values) != len(unique_values):
            diffs.append(ParameterDiff(key=key, values=values))
    return diffs


def _build_metric_summaries(
    experiment_ids: List[int],
    metrics_by_exp: Dict[int, Dict[str, float]],
    order: Literal["desc", "asc"] = "desc",
) -> List[MetricSummary]:
    all_metric_names = set()
    for exp_metrics in metrics_by_exp.values():
        all_metric_names.update(exp_metrics.keys())

    summaries: List[MetricSummary] = []
    for metric_name in sorted(all_metric_names):
        values: Dict[int, Optional[float]] = {}
        for eid in experiment_ids:
            values[eid] = metrics_by_exp[eid].get(metric_name, None)

        valid_values = {k: v for k, v in values.items() if v is not None}
        if valid_values:
            if order == "desc":
                best_exp_id = max(valid_values, key=valid_values.get)
            else:
                best_exp_id = min(valid_values, key=valid_values.get)
            best_value = valid_values[best_exp_id]
        else:
            best_exp_id = None
            best_value = None

        summaries.append(
            MetricSummary(
                metric_name=metric_name,
                values=values,
                best_experiment_id=best_exp_id,
                best_value=best_value,
            )
        )
    return summaries


@router.post("/compare", response_model=ComparisonResult)
def compare_experiments(
    experiment_ids: List[int] = Query(..., min_length=2),
    metric_order: Literal["desc", "asc"] = Query(
        default="desc",
        description="Sort order for finding best metric value (desc=higher is better, asc=lower is better)",
    ),
    db: Session = Depends(get_db),
):
    experiments = _get_experiments(db, experiment_ids)
    exp_id_order = list(experiment_ids)

    exp_infos = [
        ExperimentInfo(
            id=exp.id,
            name=exp.name,
            project=exp.project,
            status=exp.status,
            start_time=exp.start_time,
            end_time=exp.end_time,
        )
        for exp in sorted(experiments, key=lambda e: exp_id_order.index(e.id))
    ]

    params_by_exp = _collect_parameters(db, experiment_ids)
    metrics_by_exp = _collect_latest_metrics(db, experiment_ids)

    param_diffs = _build_parameter_diffs(experiment_ids, params_by_exp)
    metric_summaries = _build_metric_summaries(
        experiment_ids, metrics_by_exp, metric_order
    )

    sorted_metrics = sorted(
        metric_summaries,
        key=lambda s: (s.best_value is None, -(s.best_value or 0))
        if metric_order == "desc"
        else (s.best_value is None, s.best_value or 0),
    )

    return ComparisonResult(
        experiments=exp_infos,
        parameter_diffs=param_diffs,
        metric_summaries=metric_summaries,
        sorted_metrics=sorted_metrics,
    )


@router.get("/best")
def find_best_experiment(
    metric_name: str = Query(...),
    project: Optional[str] = Query(default=None),
    order: Literal["desc", "asc"] = Query(default="desc"),
    db: Session = Depends(get_db),
):
    subquery = (
        db.query(
            Metric.experiment_id,
            func.max(Metric.step).label("max_step"),
        )
        .filter(Metric.name == metric_name)
        .group_by(Metric.experiment_id)
        .subquery()
    )

    query = (
        db.query(
            Metric.experiment_id,
            Experiment.name.label("experiment_name"),
            Experiment.project,
            Experiment.status,
            Metric.value,
        )
        .join(Experiment, Metric.experiment_id == Experiment.id)
        .join(
            subquery,
            (Metric.experiment_id == subquery.c.experiment_id)
            & (Metric.step == subquery.c.max_step),
        )
        .filter(Metric.name == metric_name)
    )

    if project:
        query = query.filter(Experiment.project == project)

    if order == "desc":
        query = query.order_by(Metric.value.desc())
    else:
        query = query.order_by(Metric.value.asc())

    result = query.first()
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No experiments found with metric: {metric_name}",
        )

    return {
        "experiment_id": result.experiment_id,
        "experiment_name": result.experiment_name,
        "project": result.project,
        "status": result.status,
        "metric_name": metric_name,
        "metric_value": result.value,
        "order": order,
    }
