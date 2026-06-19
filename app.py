import io
import csv
import threading
import traceback
from datetime import datetime
from typing import Optional, List, Literal, Dict, Any
from pathlib import Path

from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import numpy as np

from storage import (
    init_db,
    get_db,
    SessionLocal,
    ModelVersion,
    EvaluationReport,
    BatchPredictionJob,
    FileStorage,
)

import experiments
import metrics
import models
import comparator
from models import (
    get_cached_model,
    get_cached_model_info,
    get_cached_model_classes,
    _detect_prediction_type,
)
from metrics import compute_classification_metrics, compute_regression_metrics

app = FastAPI(
    title="ML Experiment Tracker",
    description="Machine Learning Experiment Tracking & Model Management Platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/", tags=["health"])
def root():
    return {
        "name": "ML Experiment Tracker",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "healthy"}


app.include_router(experiments.router)
app.include_router(metrics.router)
app.include_router(models.router)
app.include_router(comparator.router)


# ---------------- Batch Prediction ----------------

_jobs_lock = threading.Lock()


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


def _process_batch_job(job_id: int, db_url: str, chunk_size: int = 1000):
    db = SessionLocal()
    try:
        job = db.query(BatchPredictionJob).filter(BatchPredictionJob.id == job_id).first()
        if not job:
            return
        job.status = "running"
        db.commit()

        mv = db.query(ModelVersion).filter(ModelVersion.id == job.model_id).first()
        if not mv:
            raise RuntimeError("Model version not found")

        model = get_cached_model(mv.id, mv.file_path, mv.checksum)
        info = get_cached_model_info(mv.id) or {}
        pred_type = info.get("prediction_type") or _detect_prediction_type(model)
        classes = get_cached_model_classes(mv.id)

        input_bytes = FileStorage.read_file_bytes(
            str(Path(job.input_filename)) if Path(job.input_filename).is_absolute()
            else str(Path("./dataset_storage") / job.input_filename)
        )
        if not input_bytes:
            raise RuntimeError("Input dataset file not found")

        header, data = _load_csv_to_rows(input_bytes)
        X = _rows_to_feature_matrix(header, data, feature_cols=None)

        n = X.shape[0]
        job.total_rows = n
        db.commit()

        all_preds: List[Any] = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = X[start:end]
            preds = model.predict(chunk)
            preds_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)

            if pred_type == "classification" and classes is not None:
                preds_list = [
                    classes[int(p)] if isinstance(p, (int, float, np.integer, np.floating)) else p
                    for p in preds_list
                ]

            all_preds.extend(preds_list)

            with _jobs_lock:
                job.processed_rows = end
                db.commit()

        output_buf = io.StringIO()
        writer = csv.writer(output_buf)
        out_header = list(header) + ["prediction"]
        writer.writerow(out_header)
        for i, row in enumerate(data):
            writer.writerow(list(row) + [all_preds[i]])

        result_bytes = output_buf.getvalue().encode("utf-8")
        result_path = FileStorage.save_result_file(result_bytes, job_id, "predictions.csv")

        job.result_file_path = result_path
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        db.commit()
    except Exception as e:
        job.status = "failed"
        job.error_message = f"{str(e)}\n{traceback.format_exc()}"
        job.completed_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()


class BatchJobResponse(BaseModel):
    id: int
    model_id: int
    model_name: str
    version: int
    status: str
    total_rows: int
    processed_rows: int
    result_file_path: Optional[str]
    input_filename: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


batch_router = APIRouter(prefix="/batch-predict", tags=["batch-prediction"])


@batch_router.post("", response_model=BatchJobResponse)
async def create_batch_prediction(
    model_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    mv = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        _load_csv_to_rows(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

    safe_name = Path(file.filename or f"job_{datetime.utcnow().timestamp()}.csv").name
    dataset_path = FileStorage.save_dataset_file(content, safe_name)

    job = BatchPredictionJob(
        model_id=model_id,
        model_name=mv.name,
        version=mv.version,
        status="pending",
        total_rows=0,
        processed_rows=0,
        result_file_path=None,
        input_filename=str(Path(dataset_path).name),
        error_message=None,
        created_at=datetime.utcnow(),
        completed_at=None,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    thread = threading.Thread(
        target=_process_batch_job,
        args=(job.id, ""),
        daemon=True,
    )
    thread.start()

    return BatchJobResponse(
        id=job.id,
        model_id=job.model_id,
        model_name=job.model_name,
        version=job.version,
        status=job.status,
        total_rows=job.total_rows,
        processed_rows=job.processed_rows,
        result_file_path=job.result_file_path,
        input_filename=job.input_filename,
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@batch_router.get("/{job_id}", response_model=BatchJobResponse)
def get_batch_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(BatchPredictionJob).filter(BatchPredictionJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Batch job not found")
    return BatchJobResponse(
        id=job.id,
        model_id=job.model_id,
        model_name=job.model_name,
        version=job.version,
        status=job.status,
        total_rows=job.total_rows,
        processed_rows=job.processed_rows,
        result_file_path=job.result_file_path,
        input_filename=job.input_filename,
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@batch_router.get("", response_model=List[BatchJobResponse])
def list_batch_jobs(
    model_id: Optional[int] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    query = db.query(BatchPredictionJob)
    if model_id:
        query = query.filter(BatchPredictionJob.model_id == model_id)
    if status:
        query = query.filter(BatchPredictionJob.status == status)
    jobs = query.order_by(BatchPredictionJob.created_at.desc()).limit(limit).all()
    return [
        BatchJobResponse(
            id=j.id,
            model_id=j.model_id,
            model_name=j.model_name,
            version=j.version,
            status=j.status,
            total_rows=j.total_rows,
            processed_rows=j.processed_rows,
            result_file_path=j.result_file_path,
            input_filename=j.input_filename,
            error_message=j.error_message,
            created_at=j.created_at,
            completed_at=j.completed_at,
        )
        for j in jobs
    ]


@batch_router.get("/{job_id}/download")
def download_batch_result(job_id: int, db: Session = Depends(get_db)):
    job = db.query(BatchPredictionJob).filter(BatchPredictionJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Batch job not found")
    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed yet. Current status: {job.status}",
        )
    if not job.result_file_path:
        raise HTTPException(status_code=404, detail="Result file not found")

    content = FileStorage.read_file_bytes(job.result_file_path)
    if not content:
        raise HTTPException(status_code=404, detail="Result file missing on disk")

    filename = f"batch_job_{job.id}_predictions.csv"
    return StreamingResponse(
        io.BytesIO(content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


app.include_router(batch_router)


# ---------------- Evaluation Reports ----------------

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


eval_router = APIRouter(prefix="/evaluation", tags=["evaluation"])


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


@eval_router.post("/{model_id}", response_model=EvaluationReportResponse)
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


@eval_router.get("", response_model=List[EvaluationReportResponse])
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


@eval_router.get("/{report_id}", response_model=EvaluationReportResponse)
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


@eval_router.get("/model/{model_id}/compare")
def compare_model_evaluations(
    model_id: int,
    metric_key: Optional[str] = Query(default=None, description="如 accuracy、f1、r2、rmse 等，留空则返回所有历史"),
    order: Literal["desc", "asc"] = Query(default="desc"),
    db: Session = Depends(get_db),
):
    mv = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")

    reports = (
        db.query(EvaluationReport)
        .filter(
            (EvaluationReport.model_id == model_id)
            | (EvaluationReport.model_name == mv.name)
        )
        .order_by(EvaluationReport.created_at.desc())
        .all()
    )

    summaries = []
    for r in reports:
        m = r.metrics or {}
        metric_value = None
        if metric_key:
            metric_value = m.get(metric_key)
            if metric_value is None and metric_key in ("precision", "recall", "f1"):
                metric_value = m.get(f"macro_avg_{metric_key}") or m.get(f"weighted_avg_{metric_key}")
        summaries.append(
            {
                "report_id": r.id,
                "model_id": r.model_id,
                "model_name": r.model_name,
                "version": r.version,
                "prediction_type": r.prediction_type,
                "num_samples": r.num_samples,
                "dataset_filename": r.dataset_filename,
                "metric_key": metric_key,
                "metric_value": metric_value,
                "all_metrics": m if not metric_key else None,
                "created_at": r.created_at,
            }
        )

    if metric_key:
        reverse = order == "desc"
        summaries.sort(
            key=lambda s: (s["metric_value"] is None, -(s["metric_value"] or 0) if reverse else (s["metric_value"] or 0))
        )

    return {
        "model_id": model_id,
        "model_name": mv.name,
        "metric_key": metric_key,
        "order": order,
        "count": len(summaries),
        "reports": summaries,
    }


app.include_router(eval_router)
