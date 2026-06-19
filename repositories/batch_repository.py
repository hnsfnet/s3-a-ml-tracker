from typing import Optional, List
from sqlalchemy.orm import Session

from storage import BatchPredictionJob
from .base import BaseRepository


class BatchRepository(BaseRepository[BatchPredictionJob]):
    def __init__(self, db: Session):
        super().__init__(BatchPredictionJob, db)

    def list(
        self,
        model_id: Optional[int] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[BatchPredictionJob]:
        query = self.db.query(BatchPredictionJob)
        if model_id:
            query = query.filter(BatchPredictionJob.model_id == model_id)
        if status:
            query = query.filter(BatchPredictionJob.status == status)
        return query.order_by(BatchPredictionJob.created_at.desc()).limit(limit).all()

    def update_status(self, job_id: int, **kwargs) -> Optional[BatchPredictionJob]:
        job = self.get_by_id(job_id)
        if not job:
            return None
        for key, value in kwargs.items():
            setattr(job, key, value)
        self.db.commit()
        self.db.refresh(job)
        return job

    def create_job(
        self,
        model_id: int,
        model_name: str,
        version: int,
        input_filename: str,
    ) -> BatchPredictionJob:
        job = BatchPredictionJob(
            model_id=model_id,
            model_name=model_name,
            version=version,
            status="pending",
            total_rows=0,
            processed_rows=0,
            result_file_path=None,
            input_filename=input_filename,
            error_message=None,
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job
