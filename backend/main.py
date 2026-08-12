from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.prediction import router as prediction_router
from backend.routes.fraud import router as fraud_router
from backend.routes.message import router as message_router
from backend.routes.batch import router as batch_router
from backend.services.credit_service import ensure_models

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

ensure_models()

app = FastAPI(
    title="AI Credit Card Risk & Fraud Detection",
    version="1.0.0",
    description="End-to-end demo platform for credit risk, transaction fraud and bank-message detection."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIX: Added / operator
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")

app.include_router(prediction_router, prefix="/api")
app.include_router(fraud_router, prefix="/api")
app.include_router(message_router, prefix="/api")
app.include_router(batch_router, prefix="/api")

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AI Credit Card Risk & Fraud Detection"}

@app.get("/")
def home():
    # FIX: Added / operator
    return FileResponse(FRONTEND_DIR / "dashboard.html")

@app.get("/{page}.html")
def page(page: str):
    allowed = {"dashboard", "predict", "fraud", "message", "batch", "analytics", "model"}
    if page not in allowed:
        # FIX: Added / operator
        return FileResponse(FRONTEND_DIR / "dashboard.html")
    return FileResponse(FRONTEND_DIR / f"{page}.html")
