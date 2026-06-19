import os
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
    Boolean,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./ml_tracker.db"
    MODEL_STORAGE_DIR: str = "./model_storage"

    class Config:
        env_file = ".env"


settings = Settings()

MODEL_STORAGE_PATH = Path(settings.MODEL_STORAGE_DIR)
MODEL_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    project = Column(String(255), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="running")
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    parameters = relationship("ExperimentParameter", back_populates="experiment", cascade="all, delete-orphan")
    metrics = relationship("Metric", back_populates="experiment", cascade="all, delete-orphan")
    models = relationship("ModelVersion", back_populates="experiment", cascade="all, delete-orphan")


class ExperimentParameter(Base):
    __tablename__ = "experiment_parameters"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    key = Column(String(255), nullable=False, index=True)
    value = Column(Text, nullable=False)

    experiment = relationship("Experiment", back_populates="parameters")


class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    name = Column(String(255), nullable=False, index=True)
    value = Column(Float, nullable=False)
    step = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    experiment = relationship("Experiment", back_populates="metrics")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    name = Column(String(255), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer, nullable=False)
    checksum = Column(String(64), nullable=False)
    is_production = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="models")

    __table_args__ = ()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)


class FileStorage:
    @staticmethod
    def save_model_file(file_content: bytes, experiment_id: int, model_name: str, version: int) -> tuple[str, int, str]:
        exp_dir = MODEL_STORAGE_PATH / f"exp_{experiment_id}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        file_ext = ".pkl"
        file_name = f"{model_name}_v{version}{file_ext}"
        file_path = exp_dir / file_name

        with open(file_path, "wb") as f:
            f.write(file_content)

        file_size = file_path.stat().st_size
        checksum = hashlib.sha256(file_content).hexdigest()

        return str(file_path), file_size, checksum

    @staticmethod
    def get_model_file(file_path: str) -> Optional[bytes]:
        path = Path(file_path)
        if path.exists():
            with open(path, "rb") as f:
                return f.read()
        return None

    @staticmethod
    def delete_model_file(file_path: str) -> bool:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            return True
        return False


def get_next_version(db, model_name: str) -> int:
    from sqlalchemy import func
    max_version = db.query(func.max(ModelVersion.version)).filter(ModelVersion.name == model_name).scalar()
    return (max_version or 0) + 1


def clear_production_flag(db, model_name: str):
    db.query(ModelVersion).filter(
        ModelVersion.name == model_name,
        ModelVersion.is_production == True,
    ).update({ModelVersion.is_production: False})
    db.commit()
