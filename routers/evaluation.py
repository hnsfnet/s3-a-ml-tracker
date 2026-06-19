import io
import csv
from datetime import datetime
from typing import Optional, List, Literal, Dict, Any
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import numpy as np

from storage import get_db, ModelVersion, EvaluationReport, FileStorage
from models import get_cached_model, get_cached_model_info, get_cached_model_classes, _detect_prediction_type
from calculators import compute_classification_metrics, compute_regression_metrics

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def _load_csv_to_rows(csv_content: bytes) -> tuple[List[List[str]], List[List[str]]]:
    text = csv_content.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    header = rows[0]
    data = rows[1:]
    return header, data


def _rows_to_feature_matrix(header: List[str], data: List[List[str]], feature_cols: Optional[List[str]] = None) -> np.ndarray:
    if feature_cols:
        missing = [c for c in feature_cols if c not in header]
        if missing:
            raise HTTPException(status_code=400, detail=f"Columns not found in CSV: {missing}")
        col_indices = [header.index(c) for c in feature_cols]
    else:
        col_indices = list(range(len(header)))

    matrix = []
    for i, row in enumerate(data, start=1):
        try:
            vec = [float(row[idx]) for idx in col_indices]
            matrix.append(vec)
        except (ValueError, IndexError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid numeric value at row {i}: {str(e)}",
            )
    return np.array(matrix, dtype=float)


def _resolve_prediction_type(
    explicit: Optional[str], has_predict_proba: bool, y_true: List[Any]
) -> str:
    if explicit:
        if explicit not in ("classification", "regression"):
            raise HTTPException(status_code=400, detail="prediction_type must be 'classification' or 'regression'")
        return explicit
    if has_predict_proba:
        return "classification"
    try:
        [float(v) for v in y_true]
        unique_ratio = len(set(str(v) for v in y_true)) / max(len(y_true), 1)
        if unique_ratio > 0.5:
            return "regression"
        return "classification"
    except (ValueError, TypeError):
        return "classification"


class EvaluationReportResponse(BaseModel):
    id: int
    model_id: int
    model_name: str
    version: int
    prediction_type: str
    metrics: Dict[str, Any]
    metric_descriptions: Optional[Dict[str, str]]
    num_samples: int
    dataset_filename: Optional[str]
    label_column: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


@router.post("/{model_id}", response_model=EvaluationReportResponse)
async def create_evaluation_report(
    model_id: int,
    label_column: str = Form(..., description="标签列名"),
    prediction_type: Optional[Literal["classification", "regression"]] = Form(
        default=None, description="默认自动识别"
    ),
    file: UploadFile = File(..., description="带标签的测试数据集 CSV"),
    db: Session = Depends(get_db),
):
    mv = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        header, data = _load_csv_to_rows(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

    if label_column not in header:
        raise HTTPException(
            status_code=400,
            detail=f"Label column '{label_column}' not found in CSV header: {header}",
        )

    label_idx = header.index(label_column)
    feature_cols = [h for h in header if h != label_column]
    if not feature_cols:
        raise HTTPException(status_code=400, detail="No feature columns found (only label column exists)")

    try:
        X = _rows_to_feature_matrix(header, data, feature_cols=feature_cols)
    except HTTPException:
        raise

    y_true_raw = [row[label_idx] for row in data]

    model = get_cached_model(mv.id, mv.file_path, mv.checksum)
    info = get_cached_model_info(mv.id) or {}
    classes = get_cached_model_classes(mv.id)
    has_predict_proba = "prediction_type" not in info and hasattr(model, "predict_proba")
    resolved_type = _resolve_prediction_type(
        prediction_type,
        info.get("prediction_type") == "classification" or has_predict_proba,
        y_true_raw,
    )

    preds = model.predict(X)
    y_pred = preds.tolist() if hasattr(preds, "tolist") else list(preds)

    if resolved_type == "classification" and classes is not None:
        y_pred = [
            classes[int(p)] if isinstance(p, (int, float, np.integer, np.floating)) else p
            for p in y_pred
        ]

    if resolved_type == "regression":
        try:
            y_true = [float(v) for v in y_true_raw]
            y_pred_f = [float(v) for v in y_pred]
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Regression mode requires numeric labels: {str(e)}",
            )
        metrics_dict, descriptions = compute_regression_metrics(y_true, y_pred_f)
    else:
        y_true = y_true_raw
        y_pred_s = [str(v) for v in y_pred]
        metrics_dict, descriptions = compute_classification_metrics(y_true, y_pred_s)

    report = EvaluationReport(
        model_id=model_id,
        model_name=mv.name,
        version=mv.version,
        prediction_type=resolved_type,
        metrics=metrics_dict,
        metric_descriptions=descriptions,
        num_samples=len(data),
        dataset_filename=file.filename,
        label_column=label_column,
        created_at=datetime.utcnow(),
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return EvaluationReportResponse(
        id=report.id,
        model_id=report.model_id,
        model_name=report.model_name,
        version=report.version,
        prediction_type=report.prediction_type,
        metrics=report.metrics,
        metric_descriptions=report.metric_descriptions,
        num_samples=report.num_samples,
        dataset_filename=report.dataset_filename,
        label_column=report.label_column,
        created_at=report.created_at,
    )


@router.get("", response_model=List[EvaluationReportResponse])
def list_evaluation_reports(
    model_id: Optional[int] = Query(default=None),
    model_name: Optional[str] = Query(default=None),
    prediction_type: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    query = db.query(EvaluationReport)
    if model_id:
        query = query.filter(EvaluationReport.model_id == model_id)
    if model_name:
        query = query.filter(EvaluationReport.model_name == model_name)
    if prediction_type:
        query = query.filter(EvaluationReport.prediction_type == prediction_type)
    reports = query.order_by(EvaluationReport.created_at.desc()).limit(limit).all()
    return [
        EvaluationReportResponse(
            id=r.id,
            model_id=r.model_id,
            model_name=r.model_name,
            version=r.version,
            prediction_type=r.prediction_type,
            metrics=r.metrics,
            metric_descriptions=r.metric_descriptions,
            num_samples=r.num_samples,
            dataset_filename=r.dataset_filename,
            label_column=r.label_column,
            created_at=r.created_at,
        )
        for r in reports
    ]


@router.get("/{report_id}", response_model=EvaluationReportResponse)
def get_evaluation_report(report_id: int, db: Session = Depends(get_db)):
    r = db.query(EvaluationReport).filter(EvaluationReport.id == report_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Evaluation report not found")
    return EvaluationReportResponse(
        id=r.id,
        model_id=r.model_id,
        model_name=r.model_name,
        version=r.version,
        prediction_type=r.prediction_type,
        metrics=r.metrics,
        metric_descriptions=r.metric_descriptions,
        num_samples=r.num_samples,
        dataset_filename=r.dataset_filename,
        label_column=r.label_column,
        created_at=r.created_at,
    )


@router.get("/model/{model_id}/compare")
def compare_model_evaluations(
    model_id: int,
    metric_key: Optional[str] = Query(default=None, description="按某个指标排序"),
    order: Literal["desc", "asc"] = Query(default="desc"),
    db: Session = Depends(get_db),
):
    mv = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")

    reports = (
        db.query(EvaluationReport)
        .filter(EvaluationReport.model_name == mv.name)
        .order_by(EvaluationReport.version.desc())
        .all()
    )

    if not reports:
        return {"model_name": mv.name, "reports": [], "compared_versions": []}

    def _get_metric_value(report: EvaluationReport, key: str) -> Optional[float]:
        metrics = report.metrics or {}
        if key in metrics:
            val = metrics[key]
            return float(val) if isinstance(val, (int, float)) else None
        if "per_class" in metrics and isinstance(metrics["per_class"], dict):
            for cls_val in metrics["per_class"].values():
                if isinstance(cls_val, dict) and key in cls_val:
                    return float(cls_val[key]) if isinstance(cls_val[key], (int, float)) else None
        macro_key = f"macro_avg_{key}"
        if macro_key in metrics:
            return float(metrics[macro_key]) if isinstance(metrics[macro_key], (int, float)) else None
        weighted_key = f"weighted_avg_{key}"
        if weighted_key in metrics:
            return float(metrics[weighted_key]) if isinstance(metrics[weighted_key], (int, float)) else None
        return None

    if metric_key:
        reports_with_val = [
            (r, _get_metric_value(r, metric_key))
            for r in reports
        ]
        reverse = order == "desc"
        reports_with_val.sort(
            key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0),
            reverse=reverse,
        )
        reports = [r for r, _ in reports_with_val]

    return {
        "model_name": mv.name,
        "total_reports": len(reports),
        "sort_metric": metric_key,
        "sort_order": order,
        "reports": [
            {
                "id": r.id,
                "model_id": r.model_id,
                "version": r.version,
                "prediction_type": r.prediction_type,
                "metrics": r.metrics,
                "num_samples": r.num_samples,
                "dataset_filename": r.dataset_filename,
                "created_at": r.created_at,
                "metric_value": _get_metric_value(r, metric_key) if metric_key else None,
            }
            for r in reports
        ],
    }
