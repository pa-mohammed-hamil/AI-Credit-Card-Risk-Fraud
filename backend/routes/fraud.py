
from fastapi import APIRouter
from pydantic import BaseModel, Field
import pandas as pd
from backend.services.credit_service import load_fraud

router = APIRouter()

class FraudRequest(BaseModel):
    amount: float = Field(100, ge=0)
    hour: int = Field(14, ge=0, le=23)
    distance_km: float = Field(2, ge=0)
    frequency_24h: int = Field(2, ge=0)
    new_device: int = Field(0, ge=0, le=1)
    international: int = Field(0, ge=0, le=1)
    merchant_risk: float = Field(0.2, ge=0, le=1)

@router.post("/detect-fraud")
def detect_fraud(req: FraudRequest):
    model = load_fraud()
    row = pd.DataFrame([req.model_dump()])
    prob = float(model.predict_proba(row)[0, 1])
    if prob < 0.30:
        label = "LOW"
        action = "Transaction appears low risk."
    elif prob < 0.70:
        label = "MEDIUM"
        action = "Consider additional verification."
    else:
        label = "HIGH"
        action = "Flag transaction for enhanced review."
    return {
        "fraud_score": round(prob * 100, 2),
        "risk": label,
        "suspicious": bool(prob >= 0.50),
        "recommendation": action
    }
