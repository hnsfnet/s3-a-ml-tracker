from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from storage import init_db

import experiments
import metrics
import models
import comparator
from routers import batch_predict_router, evaluation_router

app = FastAPI(
    title="ML Experiment Tracker",
    description="Machine Learning Experiment Tracking & Model Management Platform",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/", tags=["health"])
def root():
    return {
        "name": "ML Experiment Tracker",
        "version": "2.1.0",
        "status": "running",
        "docs": "/docs",
        "architecture": "repository-service-router",
    }


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "healthy"}


app.include_router(experiments.router)
app.include_router(metrics.router)
app.include_router(models.router)
app.include_router(comparator.router)
app.include_router(batch_predict_router)
app.include_router(evaluation_router)
