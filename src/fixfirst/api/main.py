"""
FastAPI application factory for FixFirst AI.

Usage (dev):
    PYTHONPATH=src uvicorn fixfirst.api.main:app --reload --port 8000

Usage (Docker, Phase 8): served by the (currently commented-out) `api`
service in docker-compose.yml once this file exists — see that file's
Phase 0 comments.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fixfirst import __version__
from fixfirst.api.routes import criticality, features, reviews
from fixfirst.api.schemas import HealthOut
from fixfirst.config.settings import settings

app = FastAPI(
    title="FixFirst AI",
    description="Automated feature prioritization engine for developers — ABSA-driven review analysis.",
    version=__version__,
)

# Permissive CORS for local dev (Streamlit dashboard, React dev server).
# Tighten allow_origins to specific domains before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_env == "development" else [],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(features.router)
app.include_router(reviews.router)
app.include_router(criticality.router)


@app.get("/health", response_model=HealthOut, tags=["health"])
def health() -> HealthOut:
    return HealthOut(status="ok", version=__version__)