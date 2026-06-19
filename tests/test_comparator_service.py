"""对比 service 层单元测试——重点覆盖分批查询合并逻辑"""
import pytest
from fastapi import HTTPException

from experiments import create_experiment, ExperimentCreate
from metrics import log_metric, MetricCreate
from comparator import (
    compare_experiments,
    find_best_experiment,
    _get_experiments,
    _collect_parameters,
    _collect_latest_metrics,
    _batch_ids,
    SQLITE_BATCH_SIZE,
)


class TestBatchIdsHelper:
    def test_batch_ids_splits_correctly(self):
        ids = list(range(1200))
        batches = _batch_ids(ids, batch_size=500)
        assert len(batches) == 3
        assert len(batches[0]) == 500
        assert len(batches[1]) == 500
        assert len(batches[2]) == 200

    def test_batch_ids_empty(self):
        assert _batch_ids([]) == []

    def test_batch_ids_single(self):
        assert _batch_ids([1, 2, 3]) == [[1, 2, 3]]


class TestComparatorBatchMerge:
    def _create_exps_with_metrics(self, db_session, count):
        ids = []
        for i in range(count):
            eid = create_experiment(
                ExperimentCreate(name=f"c{i}", project="p"), db=db_session
            ).id
            log_metric(eid, MetricCreate(name="acc", value=0.5 + i * 0.01), db=db_session)
            ids.append(eid)
        return ids

    def test_compare_many_experiments(self, db_session):
        ids = self._create_exps_with_metrics(db_session, 8)
        result = compare_experiments(
            experiment_ids=ids, metric_order="desc", db=db_session
        )
        assert len(result.experiments) == 8
        acc_summary = [s for s in result.metric_summaries if s.metric_name == "acc"][0]
        assert len(acc_summary.values) == 8
        for eid in ids:
            assert acc_summary.values[eid] is not None

    def test_compare_more_than_sqlite_limit(self, db_session):
        """超过 SQLite 参数上限（999）时验证分批不丢数据"""
        ids = self._create_exps_with_metrics(db_session, SQLITE_BATCH_SIZE + 50)
        assert len(ids) == SQLITE_BATCH_SIZE + 50

        result = compare_experiments(
            experiment_ids=ids, metric_order="desc", db=db_session
        )
        assert len(result.experiments) == SQLITE_BATCH_SIZE + 50

        acc_summary = [s for s in result.metric_summaries if s.metric_name == "acc"][0]
        non_null = [v for v in acc_summary.values.values() if v is not None]
        assert len(non_null) == SQLITE_BATCH_SIZE + 50

    def test_get_experiments_missing_raises_404(self, db_session):
        eid = create_experiment(
            ExperimentCreate(name="miss", project="p"), db=db_session
        ).id
        with pytest.raises(HTTPException) as exc:
            _get_experiments(db_session, [eid, 99999])
        assert exc.value.status_code == 404

    def test_collect_latest_metrics_batch(self, db_session):
        ids = self._create_exps_with_metrics(db_session, 3)
        result = _collect_latest_metrics(db_session, ids)
        for eid in ids:
            assert "acc" in result[eid]

    def test_compare_finds_best_experiment(self, db_session):
        ids = self._create_exps_with_metrics(db_session, 5)
        result = compare_experiments(
            experiment_ids=ids, metric_order="desc", db=db_session
        )
        acc_sorted = [s for s in result.sorted_metrics if s.metric_name == "acc"][0]
        assert acc_sorted.best_value == max(0.5 + i * 0.01 for i in range(5))

    def test_find_best_experiment(self, db_session):
        self._create_exps_with_metrics(db_session, 3)
        best = find_best_experiment(
            metric_name="acc", project=None, order="desc", db=db_session
        )
        assert best is not None
        assert best["metric_value"] == max(0.5 + i * 0.01 for i in range(3))
