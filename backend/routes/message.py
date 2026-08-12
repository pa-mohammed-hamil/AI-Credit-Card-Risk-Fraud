
from fastapi import APIRouter
from pydantic import BaseModel, Field
from backend.services.credit_service import load_message

router = APIRouter()

class MessageRequest(BaseModel):
    message: str = Field(..., min_length=3, max_length=5000)

@router.post("/detect-message")
def detect_message(req: MessageRequest):
    model = load_message()
    prob = float(model.predict_proba([req.message])[0, 1])
    if prob < 0.30:
        label = "LEGITIMATE"
        kind = "Likely legitimate"
    elif prob < 0.70:
        label = "SUSPICIOUS"
        kind = "Potentially suspicious"
    else:
        label = "FRAUD / PHISHING"
        kind = "Potential scam or phishing message"
    return {
        "classification": label,
        "risk_score": round(prob * 100, 2),
        "type": kind,
        "recommendation": "Never share OTP, PIN, CVV or passwords with anyone." if prob >= 0.30
        else "Continue to use your official banking channels."
    }
