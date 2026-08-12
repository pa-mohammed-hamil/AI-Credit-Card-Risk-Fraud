# AI Credit Card Risk & Fraud Detection

A complete local demo website with:
- Credit-card default risk prediction
- Transaction fraud scoring
- Bank/credit SMS and email-style message classification
- CSV/XLSX batch credit prediction
- Analytics dashboard
- Model performance page
- FastAPI REST API
- HTML/CSS/JavaScript frontend

## Python
Recommended: Python 3.11.9

## Run

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000

## API
- GET `/api/health`
- POST `/api/predict-credit`
- POST `/api/detect-fraud`
- POST `/api/detect-message`
- POST `/api/batch-predict`
- GET `/api/model-metrics`

The first server start automatically creates deterministic demo datasets and trains local models if model files do not exist.

## Batch CSV
Use columns:
`age,income,credit_score,employment_years,debt,credit_limit,previous_defaults,utilization,payment_history`

## Important
This is an educational/demo system. It should not be used as the sole basis for real lending, credit, fraud, or financial decisions.
