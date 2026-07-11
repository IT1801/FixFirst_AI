"""FastAPI application factory for FixFirst AI."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fixfirst import __version__
from fixfirst.api.routes import router
from fixfirst.api.schemas import HealthOut
from fixfirst.config.configuration import settings

app = FastAPI(
    title="FixFirst AI",
    description="Automated feature prioritization engine for developers — ABSA-driven review analysis.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_env == "development" else [],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health", response_model=HealthOut, tags=["health"])
def health() -> HealthOut:
    """Return a lightweight service health response."""
    return HealthOut(status="ok", version=__version__)
