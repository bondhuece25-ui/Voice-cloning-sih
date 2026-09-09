from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.health import router as health_router
from app.api.detection import router as detection_router
from app.api.auth import router as auth_router
from app.api.websocket import router as websocket_router


app = FastAPI(
    title="Voice Clone Detection API"
)


# CORS — allow the React/Vite frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API routers
app.include_router(health_router)
app.include_router(detection_router)
app.include_router(auth_router)
app.include_router(websocket_router)


@app.get("/")
def home():
    return {
        "message": "Voice Clone Detection API running"
    }


# Serve test files
TEST_DIR = Path(__file__).resolve().parents[1] / "tests"

app.mount(
    "/test",
    StaticFiles(directory=TEST_DIR, html=True),
    name="test"
)