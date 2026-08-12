
from io import BytesIO
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from backend.services.credit_service import load_credit, load_fraud, risk_label

router = APIRouter()

@router.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    content = await file.read()
    name = (file.filename or "").lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(BytesIO(content))
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Upload a CSV or Excel file.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}")

    credit_cols = ["age", "income", "credit_score", "employment_years", "debt",
                   "credit_limit", "previous_defaults", "utilization", "payment_history"]
    missing = [c for c in credit_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing)}")

    model = load_credit()
    probs = model.predict_proba(df[credit_cols])[:, 1]
    out = df.copy()
    out["default_probability"] = (probs * 100).round(2)
    out["risk"] = [risk_label(float(p)) for p in probs]

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    return StreamingResponse(
        BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=credit_risk_results.csv"}
    )
