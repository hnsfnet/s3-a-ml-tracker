from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from storage import ModelVersion, get_next_version, clear_production_flag
from .base import BaseRepository


class ModelRepository(BaseRepository[ModelVersion]):
    def __init__(self, db: Session):
        super().__init__(ModelVersion, db)

    def list(
        self,
        name: Optional[str] = None,
        experiment_id: Optional[int] = None,
        only_production: bool = False,
    ) -> List[ModelVersion]:
        query = self.db.query(ModelVersion)
        if name:
            query = query.filter(ModelVersion.name == name)
        if experiment_id:
            query = query.filter(ModelVersion.experiment_id == experiment_id)
        if only_production:
            query = query.filter(ModelVersion.is_production == True)
        return query.order_by(ModelVersion.name, ModelVersion.version.desc()).all()

    def list_names(self) -> List[str]:
        names = self.db.query(ModelVersion.name).distinct().all()
        return [n[0] for n in names]

    def get_by_name(self, name: str, production: bool = False) -> Optional[ModelVersion]:
        query = self.db.query(ModelVersion).filter(ModelVersion.name == name)
        if production:
            query = query.filter(ModelVersion.is_production == True)
            return query.first()
        return query.order_by(ModelVersion.version.desc()).first()

    def get_production(self, model_name: str) -> Optional[ModelVersion]:
        return (
            self.db.query(ModelVersion)
            .filter(ModelVersion.name == model_name, ModelVersion.is_production == True)
            .first()
        )

    def get_next_version(self, model_name: str) -> int:
        return get_next_version(self.db, model_name)

    def set_production(self, model_id: int) -> Optional[ModelVersion]:
        mv = self.get_by_id(model_id)
        if not mv:
            return None
        clear_production_flag(self.db, mv.name)
        mv.is_production = True
        self.db.commit()
        self.db.refresh(mv)
        return mv

    def create_version(
        self,
        experiment_id: int,
        name: str,
        version: int,
        file_path: str,
        file_size: int,
        checksum: str,
    ) -> ModelVersion:
        mv = ModelVersion(
            experiment_id=experiment_id,
            name=name,
            version=version,
            file_path=file_path,
            file_size=file_size,
            checksum=checksum,
            is_production=False,
        )
        self.db.add(mv)
        self.db.commit()
        self.db.refresh(mv)
        return mv
