
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from backend.services.credit_service import load_credit, risk_label, metrics

router = APIRouter()

class CreditRequest(BaseModel):
    age: int = Field(30, ge=18, le=100)
    income: float = Field(60000, ge=0)
    credit_score: float = Field(700, ge=300, le=850)
    employment_years: float = Field(5, ge=0)
    debt: float = Field(10000, ge=0)
    credit_limit: float = Field(30000, gt=0)
    previous_defaults: int = Field(0, ge=0, le=10)
    utilization: float = Field(0.30, ge=0, le=1)
    payment_history: float = Field(90, ge=0, le=100)

@router.post("/predict-credit")
def predict_credit(req: CreditRequest):
    model = load_credit()
    row = pd.DataFrame([req.model_dump()])
    prob = float(model.predict_proba(row)[0, 1])
    label = risk_label(prob)
    recommendation = {
        "LOW": "Eligible for standard review",
        "MEDIUM": "Manual review recommended",
        "HIGH": "High-risk application; enhanced review recommended"
    }[label]
    return {
        "risk": label,
        "default_probability": round(prob * 100, 2),
        "recommendation": recommendation
    }

@router.get("/model-metrics")
def model_metrics():
    return metrics()
