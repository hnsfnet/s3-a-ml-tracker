from typing import Optional, List
from sqlalchemy.orm import Session

from storage import EvaluationReport
from .base import BaseRepository


class EvaluationRepository(BaseRepository[EvaluationReport]):
    def __init__(self, db: Session):
        super().__init__(EvaluationReport, db)

    def list(
        self,
        model_id: Optional[int] = None,
        model_name: Optional[str] = None,
        prediction_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[EvaluationReport]:
        query = self.db.query(EvaluationReport)
        if model_id:
            query = query.filter(EvaluationReport.model_id == model_id)
        if model_name:
            query = query.filter(EvaluationReport.model_name == model_name)
        if prediction_type:
            query = query.filter(EvaluationReport.prediction_type == prediction_type)
        return query.order_by(EvaluationReport.created_at.desc()).limit(limit).all()

    def list_by_model_name(
        self,
        model_id: int,
    ) -> List[EvaluationReport]:
        mv = self.db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
        if not mv:
            return []
        return (
            self.db.query(EvaluationReport)
            .filter(EvaluationReport.model_name == mv.name)
            .order_by(EvaluationReport.version.desc())
            .all()
        )


from storage import ModelVersion
