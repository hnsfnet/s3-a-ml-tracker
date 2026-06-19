import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from storage import Base, get_db
import app as app_module


@pytest.fixture(scope="function")
def db_engine():
    """每个测试函数独立的内存 SQLite 引擎，完全隔离"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """每个测试独立的数据库会话，测试结束自动回滚"""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=db_engine
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="function")
def client():
    """TestClient，使用 StaticPool 内存数据库以便 ASGI 工作线程共享同一连接"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )

    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app_module.app.dependency_overrides[get_db] = override_get_db
    with TestClient(app_module.app) as c:
        yield c
    app_module.app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def sample_experiment_id(client):
    """创建一个示例实验，返回 ID"""
    resp = client.post(
        "/experiments",
        json={"name": "test-exp", "project": "test-project"},
    )
    return resp.json()["id"]


@pytest.fixture
def production_model_id(client, sample_experiment_id):
    """注册并标记一个 sklearn 模型为生产可用，返回模型 ID"""
    import pickle
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit([[0, 0], [1, 1]], [0, 1])
    content = pickle.dumps(model)

    resp = client.post(
        f"/models/register-sklearn/{sample_experiment_id}",
        data={"name": "test-model"},
        files={"file": ("model.pkl", content, "application/octet-stream")},
    )
    model_id = resp.json()["id"]
    client.post(f"/models/{model_id}/production")
    return model_id
