import json
from datetime import datetime
from typing import Optional, List, Literal, Dict, Any, Tuple
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func
import numpy as np

from storage import get_db, Experiment, Metric
from calculators import (
    compute_classification_metrics as _calc_classification,
    compute_regression_metrics as _calc_regression,
    MetricCalculatorRegistry,
)

router = APIRouter(prefix="/metrics", tags=["metrics"])


CLASSIFICATION_METRIC_DESCRIPTIONS: Dict[str, str] = {
    "accuracy": "准确率（Accuracy）：预测正确的样本数占总样本数的比例，越接近1越好。适用于类别均衡的场景。",
    "precision": "精确率（Precision）：被预测为正类的样本中真正是正类的比例，越接近1越好。用于关注误报代价高的场景（如垃圾邮件过滤）。",
    "recall": "召回率（Recall）：真正是正类的样本中被预测为正类的比例，越接近1越好。用于关注漏报代价高的场景（如疾病筛查）。",
    "f1": "F1 分数：精确率和召回率的调和平均值，越接近1越好。综合评估模型在精确率和召回率上的表现，类别不均衡时更可靠。",
    "confusion_matrix": "混淆矩阵：行代表真实标签，列代表预测标签。对角线元素是预测正确的数量，非对角线是预测错误的数量，可直观看到模型易混淆的类别。",
    "support": "样本数（Support）：每个类别在数据集中的实际样本数量。",
    "macro_avg_precision": "宏平均精确率：对每个类别计算精确率后取算术平均，不考虑样本量差异，适合评估模型对每个类别的整体表现。",
    "macro_avg_recall": "宏平均召回率：对每个类别计算召回率后取算术平均。",
    "macro_avg_f1": "宏平均 F1：对每个类别计算 F1 后取算术平均。",
    "weighted_avg_precision": "加权平均精确率：按每类样本量加权计算精确率，考虑类别不均衡。",
    "weighted_avg_recall": "加权平均召回率：按每类样本量加权计算召回率。",
    "weighted_avg_f1": "加权平均 F1：按每类样本量加权计算 F1。",
}

REGRESSION_METRIC_DESCRIPTIONS: Dict[str, str] = {
    "mse": "均方误差（MSE）：预测值与真实值之差的平方的平均值，越小越好。对较大误差惩罚更重。",
    "mae": "平均绝对误差（MAE）：预测值与真实值之差的绝对值的平均值，越小越好。对异常值不敏感。",
    "rmse": "均方根误差（RMSE）：MSE 的平方根，与目标变量单位一致，越小越好。直观反映预测偏差大小。",
    "r2": "决定系数（R²）：衡量模型对数据变异性的解释能力，取值越接近1越好。1 表示完美预测，0 表示等价于均值预测，负数表示比均值还差。",
    "explained_variance": "解释方差：模型解释的方差占总方差的比例，越接近1越好。",
    "mape": "平均绝对百分比误差（MAPE）：预测误差占真实值百分比的平均值，越小越好。便于跨业务场景理解误差相对大小。",
}


def _ensure_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"scikit-learn is required for evaluation: {str(e)}",
        )


def compute_classification_metrics(
    y_true: List[Any], y_pred: List[Any]
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    _ensure_sklearn()
    return _calc_classification(y_true, y_pred)


def compute_regression_metrics(
    y_true: List[float], y_pred: List[float]
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    _ensure_sklearn()
    return _calc_regression(y_true, y_pred)


class MetricCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    value: float
    step: Optional[int] = Field(default=None)


class MetricResponse(BaseModel):
    id: int
    experiment_id: int
    name: str
    value: float
    step: int
    timestamp: datetime

    class Config:
        from_attributes = True


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


def _get_next_step(db: Session, experiment_id: int, metric_name: str) -> int:
    from sqlalchemy import func
    max_step = db.query(func.max(Metric.step)).filter(
        Metric.experiment_id == experiment_id,
        Metric.name == metric_name,
    ).scalar()
    return (max_step if max_step is not None else -1) + 1


@router.post("/{experiment_id}", response_model=MetricResponse)
def log_metric(
    experiment_id: int,
    metric_create: MetricCreate,
    db: Session = Depends(get_db),
):
    _check_experiment(db, experiment_id)
    step = metric_create.step
    if step is None:
        step = _get_next_step(db, experiment_id, metric_create.name)
    metric = Metric(
        experiment_id=experiment_id,
        name=metric_create.name,
        value=metric_create.value,
        step=step,
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
    next_steps = {}
    for item in metrics_batch:
        step = item.step
        if step is None:
            if item.name not in next_steps:
                next_steps[item.name] = _get_next_step(db, experiment_id, item.name)
            step = next_steps[item.name]
            next_steps[item.name] += 1
        metric = Metric(
            experiment_id=experiment_id,
            name=item.name,
            value=item.value,
            step=step,
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
