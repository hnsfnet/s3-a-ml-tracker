"""模型 service 层单元测试——重点覆盖版本号递增和并发注册冲突"""
import pickle
import threading
import pytest
from fastapi import HTTPException

from experiments import create_experiment, ExperimentCreate
from models import register_sklearn_model, list_models
from storage import ModelVersion, get_next_version
import storage


def _make_sklearn_bytes():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit([[0, 0], [1, 1]], [0, 1])
    return pickle.dumps(model)


class _FakeUploadFile:
    def __init__(self, content, filename="model.pkl"):
        self._content = content
        self.filename = filename

    async def read(self):
        return self._content


class TestModelVersionIncrement:
    @pytest.mark.asyncio
    async def test_version_auto_increment(self, db_session, monkeypatch, tmp_path):
        monkeypatch.setattr(storage, "MODEL_STORAGE_PATH", tmp_path)
        eid = create_experiment(
            ExperimentCreate(name="v1", project="p"), db=db_session
        ).id
        content = _make_sklearn_bytes()

        m1 = await register_sklearn_model(
            eid, name="clf", file=_FakeUploadFile(content), db=db_session
        )
        m2 = await register_sklearn_model(
            eid, name="clf", file=_FakeUploadFile(content), db=db_session
        )
        m3 = await register_sklearn_model(
            eid, name="clf", file=_FakeUploadFile(content), db=db_session
        )
        assert m1.version == 1
        assert m2.version == 2
        assert m3.version == 3

    @pytest.mark.asyncio
    async def test_different_models_independent_versioning(self, db_session, monkeypatch, tmp_path):
        monkeypatch.setattr(storage, "MODEL_STORAGE_PATH", tmp_path)
        eid = create_experiment(
            ExperimentCreate(name="v2", project="p"), db=db_session
        ).id
        content = _make_sklearn_bytes()

        m_a = await register_sklearn_model(
            eid, name="model_a", file=_FakeUploadFile(content), db=db_session
        )
        m_b = await register_sklearn_model(
            eid, name="model_b", file=_FakeUploadFile(content), db=db_session
        )
        m_a2 = await register_sklearn_model(
            eid, name="model_a", file=_FakeUploadFile(content), db=db_session
        )
        assert m_a.version == 1
        assert m_b.version == 1
        assert m_a2.version == 2

    def test_unique_constraint_prevents_duplicate(self, db_session):
        eid = create_experiment(
            ExperimentCreate(name="v3", project="p"), db=db_session
        ).id
        first = ModelVersion(
            experiment_id=eid, name="dup", version=1,
            file_path="x", file_size=1, checksum="c",
        )
        db_session.add(first)
        db_session.commit()

        dup = ModelVersion(
            experiment_id=eid, name="dup", version=1,
            file_path="y", file_size=2, checksum="d",
        )
        db_session.add(dup)
        from sqlalchemy.exc import IntegrityError
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_get_next_version_returns_correct(self, db_session):
        assert get_next_version(db_session, "no_exist") == 1
        eid = create_experiment(
            ExperimentCreate(name="vn", project="p"), db=db_session
        ).id
        mv = ModelVersion(
            experiment_id=eid, name="seq", version=5,
            file_path="x", file_size=1, checksum="c",
        )
        db_session.add(mv)
        db_session.commit()
        assert get_next_version(db_session, "seq") == 6


class TestModelConcurrentRegistration:
    def test_concurrent_registration_no_duplicate_version(self, db_engine, monkeypatch, tmp_path):
        """多线程并发注册同一模型名，验证版本号不重复"""
        monkeypatch.setattr(storage, "MODEL_STORAGE_PATH", tmp_path)
        from sqlalchemy.orm import sessionmaker
        TestSession = sessionmaker(bind=db_engine)

        session = TestSession()
        eid = create_experiment(
            ExperimentCreate(name="conc", project="p"), db=session
        ).id
        session.close()

        results = []
        barrier = threading.Barrier(5)

        def register():
            try:
                barrier.wait(timeout=5)
                import asyncio
                loop = asyncio.new_event_loop()
                sess = TestSession()
                try:
                    content = _make_sklearn_bytes()
                    m = loop.run_until_complete(
                        register_sklearn_model(
                            eid,
                            name="concurrent_model",
                            file=_FakeUploadFile(content),
                            db=sess,
                        )
                    )
                    results.append(m.version)
                finally:
                    sess.close()
                    loop.close()
            except Exception as e:
                results.append(("error", str(e)))

        threads = [threading.Thread(target=register) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        successful_versions = [v for v in results if isinstance(v, int)]
        assert len(successful_versions) == len(set(successful_versions)), (
            f"Duplicate versions found: {sorted(successful_versions)}"
        )
