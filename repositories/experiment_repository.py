from typing import Optional, List, Dict, Any
from sqlalchemy import func, desc as sql_desc, asc as sql_asc
from sqlalchemy.orm import Session

from storage import Experiment, ExperimentTag, Metric
from .base import BaseRepository


class ExperimentRepository(BaseRepository[Experiment]):
    def __init__(self, db: Session):
        super().__init__(Experiment, db)

    def list(
        self,
        project: Optional[str] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Experiment]:
        query = self.db.query(Experiment)
        if project:
            query = query.filter(Experiment.project == project)
        if status:
            query = query.filter(Experiment.status == status)
        return query.order_by(Experiment.created_at.desc()).offset(skip).limit(limit).all()

    def get_by_name(self, name: str) -> Optional[Experiment]:
        return self.db.query(Experiment).filter(Experiment.name == name).first()

    def search(
        self,
        project: Optional[str] = None,
        status: Optional[str] = None,
        name_contains: Optional[str] = None,
        tags: Optional[List[str]] = None,
        tag_mode: str = "any",
        created_from: Optional[Any] = None,
        created_to: Optional[Any] = None,
        sort_by_metric: Optional[str] = None,
        sort_order: str = "desc",
        skip: int = 0,
        limit: int = 100,
    ) -> List[Experiment]:
        query = self.db.query(Experiment)

        if project:
            query = query.filter(Experiment.project == project)
        if status:
            query = query.filter(Experiment.status == status)
        if name_contains:
            query = query.filter(Experiment.name.like(f"%{name_contains}%"))
        if created_from:
            query = query.filter(Experiment.created_at >= created_from)
        if created_to:
            query = query.filter(Experiment.created_at <= created_to)

        if tags:
            if tag_mode == "all":
                for tag in tags:
                    subq = (
                        self.db.query(ExperimentTag.experiment_id)
                        .filter(ExperimentTag.tag == tag)
                        .subquery()
                    )
                    query = query.filter(Experiment.id.in_(subq))
            else:
                subq = (
                    self.db.query(ExperimentTag.experiment_id)
                    .filter(ExperimentTag.tag.in_(tags))
                    .subquery()
                )
                query = query.filter(Experiment.id.in_(subq))

        if sort_by_metric:
            order_func = sql_desc if sort_order == "desc" else sql_asc
            subquery = (
                self.db.query(
                    Metric.experiment_id,
                    func.max(Metric.step).label("max_step"),
                )
                .filter(Metric.name == sort_by_metric)
                .group_by(Metric.experiment_id)
                .subquery()
            )
            metric_subq = (
                self.db.query(Metric.experiment_id, Metric.value)
                .join(
                    subquery,
                    (Metric.experiment_id == subquery.c.experiment_id)
                    & (Metric.step == subquery.c.max_step),
                )
                .filter(Metric.name == sort_by_metric)
                .subquery()
            )
            query = query.outerjoin(
                metric_subq, Experiment.id == metric_subq.c.experiment_id
            )
            query = query.order_by(
                metric_subq.c.value.is_(None),
                order_func(metric_subq.c.value) if sort_order == "desc" else sql_asc(metric_subq.c.value),
            )
        else:
            query = query.order_by(Experiment.created_at.desc())

        return query.offset(skip).limit(limit).all()

    def get_tags(self, experiment_id: int) -> List[str]:
        tags = (
            self.db.query(ExperimentTag)
            .filter(ExperimentTag.experiment_id == experiment_id)
            .all()
        )
        return sorted([t.tag for t in tags])

    def add_tags(self, experiment_id: int, tags: List[str]) -> List[str]:
        existing = set(self.get_tags(experiment_id))
        for tag in tags:
            if tag not in existing:
                self.db.add(ExperimentTag(experiment_id=experiment_id, tag=tag))
                existing.add(tag)
        self.db.commit()
        return sorted(list(existing))

    def remove_tags(self, experiment_id: int, tags: List[str]) -> List[str]:
        tag_set = set(tags)
        self.db.query(ExperimentTag).filter(
            ExperimentTag.experiment_id == experiment_id,
            ExperimentTag.tag.in_(tags),
        ).delete(synchronize_session=False)
        self.db.commit()
        return self.get_tags(experiment_id)

    def list_all_tags(self, project: Optional[str] = None) -> List[str]:
        query = self.db.query(ExperimentTag.tag).distinct()
        if project:
            query = query.join(Experiment).filter(Experiment.project == project)
        tags = query.all()
        return sorted([t[0] for t in tags])

    def get_parameters(self, experiment_id: int) -> Dict[str, str]:
        from storage import ExperimentParameter
        params = (
            self.db.query(ExperimentParameter)
            .filter(ExperimentParameter.experiment_id == experiment_id)
            .all()
        )
        return {p.key: p.value for p in params}
