"""指标 service 层单元测试——重点覆盖 step 自动递增和按步数排序"""
import pytest
from datetime import datetime
from fastapi import HTTPException

from metrics import (
    log_metric,
    log_metrics_batch,
    get_experiment_metrics,
    get_metric_names,
    get_metric_ranking,
    MetricCreate,
)
from experiments import create_experiment, ExperimentCreate


class TestMetricStepAutoIncrement:
    def test_step_auto_increments_when_not_provided(self, db_session):
        eid = create_experiment(
            ExperimentCreate(name="m1", project="p"), db=db_session
        ).id
        m1 = log_metric(eid, MetricCreate(name="loss", value=0.5), db=db_session)
        m2 = log_metric(eid, MetricCreate(name="loss", value=0.3), db=db_session)
        m3 = log_metric(eid, MetricCreate(name="loss", value=0.1), db=db_session)
        assert m1.step == 0
        assert m2.step == 1
        assert m3.step == 2

    def test_explicit_step_preserved(self, db_session):
        eid = create_experiment(
            ExperimentCreate(name="m2", project="p"), db=db_session
        ).id
        m1 = log_metric(eid, MetricCreate(name="acc", value=0.8, step=10), db=db_session)
        m2 = log_metric(eid, MetricCreate(name="acc", value=0.9, step=20), db=db_session)
        assert m1.step == 10
        assert m2.step == 20

    def test_batch_step_auto_increment(self, db_session):
        eid = create_experiment(
            ExperimentCreate(name="m3", project="p"), db=db_session
        ).id
        result = log_metrics_batch(
            eid,
            [
                MetricCreate(name="loss", value=1.0),
                MetricCreate(name="loss", value=0.8),
                MetricCreate(name="loss", value=0.6),
                MetricCreate(name="acc", value=0.5),
            ],
            db=db_session,
        )
        steps = {m.name: m.step for m in result}
        assert steps["loss"] in [0, 1, 2]
        assert steps["acc"] == 0


class TestMetricQueryOrdering:
    def test_metrics_ordered_by_step(self, db_session):
        eid = create_experiment(
            ExperimentCreate(name="m4", project="p"), db=db_session
        ).id
        log_metric(eid, MetricCreate(name="loss", value=0.5, step=2), db=db_session)
        log_metric(eid, MetricCreate(name="loss", value=0.1, step=0), db=db_session)
        log_metric(eid, MetricCreate(name="loss", value=0.3, step=1), db=db_session)

        result = get_experiment_metrics(eid, metric_name="loss", db=db_session)
        values = [m.value for m in result]
        assert values == [0.1, 0.3, 0.5]

    def test_multiple_metric_names_ordered(self, db_session):
        eid = create_experiment(
            ExperimentCreate(name="m5", project="p"), db=db_session
        ).id
        log_metric(eid, MetricCreate(name="acc", value=0.9, step=0), db=db_session)
        log_metric(eid, MetricCreate(name="loss", value=0.5, step=0), db=db_session)
        log_metric(eid, MetricCreate(name="loss", value=0.3, step=1), db=db_session)
        log_metric(eid, MetricCreate(name="acc", value=0.95, step=1), db=db_session)

        result = get_experiment_metrics(eid, metric_name=None, db=db_session)
        names = [m.name for m in result]
        assert names == sorted(names)

    def test_get_metric_names(self, db_session):
        eid = create_experiment(
            ExperimentCreate(name="m6", project="p"), db=db_session
        ).id
        log_metric(eid, MetricCreate(name="loss", value=0.5), db=db_session)
        log_metric(eid, MetricCreate(name="acc", value=0.9), db=db_session)
        names = get_metric_names(eid, db=db_session)
        assert set(names) == {"loss", "acc"}


class TestMetricRanking:
    def test_ranking_desc(self, db_session):
        for i, val in enumerate([0.9, 0.5, 0.7]):
            eid = create_experiment(
                ExperimentCreate(name=f"r{i}", project="p"), db=db_session
            ).id
            log_metric(eid, MetricCreate(name="acc", value=val), db=db_session)
        ranking = get_metric_ranking("acc", project=None, order="desc", limit=50, db=db_session)
        values = [r.latest_value for r in ranking]
        assert values == sorted(values, reverse=True)

    def test_ranking_asc(self, db_session):
        for i, val in enumerate([0.9, 0.5, 0.7]):
            eid = create_experiment(
                ExperimentCreate(name=f"r2_{i}", project="p"), db=db_session
            ).id
            log_metric(eid, MetricCreate(name="acc", value=val), db=db_session)
        ranking = get_metric_ranking("acc", project=None, order="asc", limit=50, db=db_session)
        values = [r.latest_value for r in ranking]
        assert values == sorted(values)

    def test_ranking_uses_latest_step(self, db_session):
        eid = create_experiment(
            ExperimentCreate(name="r3", project="p"), db=db_session
        ).id
        log_metric(eid, MetricCreate(name="loss", value=0.9, step=0), db=db_session)
        log_metric(eid, MetricCreate(name="loss", value=0.1, step=1), db=db_session)
        ranking = get_metric_ranking("loss", project=None, order="desc", limit=50, db=db_session)
        assert ranking[0].latest_value == 0.1
        assert ranking[0].latest_step == 1
