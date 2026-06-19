from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy import func, desc as sql_desc, asc as sql_asc
from sqlalchemy.orm import Session

from storage import Metric, Experiment
from .base import BaseRepository

SQLITE_BATCH_SIZE = 500


class MetricRepository(BaseRepository[Metric]):
    def __init__(self, db: Session):
        super().__init__(Metric, db)

    def get_by_experiment(
        self,
        experiment_id: int,
        metric_name: Optional[str] = None,
    ) -> List[Metric]:
        query = self.db.query(Metric).filter(Metric.experiment_id == experiment_id)
        if metric_name:
            query = query.filter(Metric.name == metric_name)
        return query.order_by(Metric.name, Metric.step, Metric.timestamp).all()

    def list_names(self, experiment_id: int) -> List[str]:
        names = (
            self.db.query(Metric.name)
            .filter(Metric.experiment_id == experiment_id)
            .distinct()
            .all()
        )
        return [n[0] for n in names]

    def get_next_step(self, experiment_id: int, metric_name: str) -> int:
        max_step = (
            self.db.query(func.max(Metric.step))
            .filter(Metric.experiment_id == experiment_id, Metric.name == metric_name)
            .scalar()
        )
        return (max_step or -1) + 1

    def get_ranking(
        self,
        metric_name: str,
        project: Optional[str] = None,
        order: str = "desc",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        order_func = sql_desc if order == "desc" else sql_asc

        subquery = (
            self.db.query(
                Metric.experiment_id,
                Metric.name,
                func.max(Metric.step).label("max_step"),
            )
            .filter(Metric.name == metric_name)
            .group_by(Metric.experiment_id, Metric.name)
            .subquery()
        )

        query = (
            self.db.query(
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
        return query.all()

    def get_latest_by_experiments(
        self, experiment_ids: List[int]
    ) -> Dict[int, Dict[str, float]]:
        result: Dict[int, Dict[str, float]] = {eid: {} for eid in experiment_ids}

        batches = [
            experiment_ids[i:i + SQLITE_BATCH_SIZE]
            for i in range(0, len(experiment_ids), SQLITE_BATCH_SIZE)
        ]

        for batch in batches:
            subquery = (
                self.db.query(
                    Metric.experiment_id,
                    Metric.name,
                    func.max(Metric.step).label("max_step"),
                )
                .filter(Metric.experiment_id.in_(batch))
                .group_by(Metric.experiment_id, Metric.name)
                .subquery()
            )

            metrics = (
                self.db.query(Metric)
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

    def find_best(
        self,
        metric_name: str,
        project: Optional[str] = None,
        order: str = "desc",
    ) -> Optional[Dict[str, Any]]:
        subquery = (
            self.db.query(
                Metric.experiment_id,
                func.max(Metric.step).label("max_step"),
            )
            .filter(Metric.name == metric_name)
            .group_by(Metric.experiment_id)
            .subquery()
        )

        query = (
            self.db.query(
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

        return query.first()
