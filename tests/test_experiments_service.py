"""实验 service 层单元测试——重点覆盖状态流转规则"""
import pytest
from fastapi import HTTPException

from experiments import (
    create_experiment,
    update_experiment,
    get_experiment,
    list_experiments,
    delete_experiment,
    ExperimentCreate,
    ExperimentUpdate,
    ALLOWED_TRANSITIONS,
)


class TestExperimentCRUD:
    def test_create_experiment_defaults_running(self, db_session):
        exp = create_experiment(
            ExperimentCreate(name="exp-1", project="proj-a"), db=db_session
        )
        assert exp.id is not None
        assert exp.status == "running"
        assert exp.start_time is not None
        assert exp.end_time is None

    def test_create_with_parameters_and_tags(self, db_session):
        exp = create_experiment(
            ExperimentCreate(
                name="exp-2",
                project="proj-a",
                parameters={"lr": "0.01", "epochs": "100"},
                tags=["baseline", "v1"],
            ),
            db=db_session,
        )
        assert len(exp.parameters) == 2
        assert exp.tags == ["baseline", "v1"]

    def test_get_experiment_not_found(self, db_session):
        with pytest.raises(HTTPException) as exc:
            get_experiment(99999, db=db_session)
        assert exc.value.status_code == 404

    def test_list_filter_by_project(self, db_session):
        create_experiment(ExperimentCreate(name="a", project="p1"), db=db_session)
        create_experiment(ExperimentCreate(name="b", project="p2"), db=db_session)
        result = list_experiments(project="p1", status=None, skip=0, limit=100, db=db_session)
        assert len(result) == 1
        assert result[0].project == "p1"

    def test_delete_experiment(self, db_session):
        exp = create_experiment(
            ExperimentCreate(name="del", project="p"), db=db_session
        )
        delete_experiment(exp.id, db=db_session)
        with pytest.raises(HTTPException) as exc:
            get_experiment(exp.id, db=db_session)
        assert exc.value.status_code == 404


class TestExperimentStateTransitions:
    """状态流转规则：running→success/failed 允许；终态不可回退、不可互转"""

    def _create(self, db_session):
        return create_experiment(
            ExperimentCreate(name="state-exp", project="p"), db=db_session
        )

    def test_running_to_success_allowed(self, db_session):
        exp = self._create(db_session)
        updated = update_experiment(
            exp.id, ExperimentUpdate(status="success"), db=db_session
        )
        assert updated.status == "success"
        assert updated.end_time is not None

    def test_running_to_failed_allowed(self, db_session):
        exp = self._create(db_session)
        updated = update_experiment(
            exp.id, ExperimentUpdate(status="failed"), db=db_session
        )
        assert updated.status == "failed"
        assert updated.end_time is not None

    def test_failed_cannot_revert_to_running(self, db_session):
        exp = self._create(db_session)
        update_experiment(
            exp.id, ExperimentUpdate(status="failed"), db=db_session
        )
        with pytest.raises(HTTPException) as exc:
            update_experiment(
                exp.id, ExperimentUpdate(status="running"), db=db_session
            )
        assert exc.value.status_code == 400
        assert "Cannot transition" in exc.value.detail

    def test_success_cannot_revert_to_running(self, db_session):
        exp = self._create(db_session)
        update_experiment(
            exp.id, ExperimentUpdate(status="success"), db=db_session
        )
        with pytest.raises(HTTPException) as exc:
            update_experiment(
                exp.id, ExperimentUpdate(status="running"), db=db_session
            )
        assert exc.value.status_code == 400

    def test_failed_cannot_change_to_success(self, db_session):
        exp = self._create(db_session)
        update_experiment(
            exp.id, ExperimentUpdate(status="failed"), db=db_session
        )
        with pytest.raises(HTTPException) as exc:
            update_experiment(
                exp.id, ExperimentUpdate(status="success"), db=db_session
            )
        assert exc.value.status_code == 400

    def test_success_cannot_change_to_failed(self, db_session):
        exp = self._create(db_session)
        update_experiment(
            exp.id, ExperimentUpdate(status="success"), db=db_session
        )
        with pytest.raises(HTTPException) as exc:
            update_experiment(
                exp.id, ExperimentUpdate(status="failed"), db=db_session
            )
        assert exc.value.status_code == 400

    def test_invalid_status_rejected(self, db_session):
        exp = self._create(db_session)
        with pytest.raises(HTTPException) as exc:
            update_experiment(
                exp.id, ExperimentUpdate(status="paused"), db=db_session
            )
        assert exc.value.status_code == 400

    def test_idempotent_status_update(self, db_session):
        """更新为相同状态应允许（幂等）"""
        exp = self._create(db_session)
        updated = update_experiment(
            exp.id, ExperimentUpdate(status="running"), db=db_session
        )
        assert updated.status == "running"
