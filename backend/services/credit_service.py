
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "backend" / "models"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

CREDIT_FEATURES = [
    "age", "income", "credit_score", "employment_years", "debt",
    "credit_limit", "previous_defaults", "utilization", "payment_history"
]

def _metrics(y, pred, prob):
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, prob)), 4),
    }

def _make_credit_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, n)
    income = rng.normal(65000, 22000, n).clip(18000, 180000)
    credit_score = rng.normal(690, 70, n).clip(300, 850)
    employment_years = rng.integers(0, 35, n)
    debt = rng.normal(18000, 12000, n).clip(500, 100000)
    credit_limit = rng.normal(35000, 16000, n).clip(3000, 120000)
    previous_defaults = rng.poisson(0.35, n).clip(0, 4)
    utilization = rng.beta(2.2, 3.0, n)
    payment_history = rng.normal(82, 12, n).clip(35, 100)

    risk = (
        0.00002 * debt
        - 0.010 * (credit_score - 650)
        + 1.8 * utilization
        + 0.65 * previous_defaults
        - 0.018 * payment_history
        - 0.000006 * income
        + rng.normal(0, 0.8, n)
    )
    default = (risk > np.median(risk)).astype(int)

    df = pd.DataFrame({
        "age": age, "income": income.round(2), "credit_score": credit_score.round(0),
        "employment_years": employment_years, "debt": debt.round(2),
        "credit_limit": credit_limit.round(2), "previous_defaults": previous_defaults,
        "utilization": utilization.round(4), "payment_history": payment_history.round(2),
        "default": default
    })
    return df

def _make_fraud_data(n=1600, seed=43):
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(4.2, 1.0, n).clip(2, 15000)
    hour = rng.integers(0, 24, n)
    distance = rng.exponential(15, n).clip(0, 500)
    frequency = rng.poisson(3, n).clip(0, 30)
    new_device = rng.integers(0, 2, n)
    international = rng.integers(0, 2, n)
    merchant_risk = rng.random(n)

    score = (
        0.00035 * amount
        + 0.9 * (hour < 5)
        + 0.025 * distance
        + 0.25 * frequency
        + 1.0 * new_device
        + 0.8 * international
        + 1.5 * merchant_risk
        + rng.normal(0, 0.5, n)
    )
    fraud = (score > np.quantile(score, 0.84)).astype(int)

    return pd.DataFrame({
        "amount": amount.round(2), "hour": hour, "distance_km": distance.round(2),
        "frequency_24h": frequency, "new_device": new_device,
        "international": international, "merchant_risk": merchant_risk.round(3),
        "fraud": fraud
    })

def _make_messages():
    legitimate = [
        "Your card payment of 2500 was successful.",
        "Your credit card statement is now available.",
        "Payment received. Thank you for using your credit card.",
        "Your bank transfer was completed successfully.",
        "Your card ending 4821 was used for a purchase of 1200.",
        "Your monthly credit card bill is due on 15 August.",
        "Your transaction of 850 has been approved.",
        "Your account statement is ready to view in the official banking app.",
    ]
    fraud = [
        "Urgent! Your bank account will be blocked today. Verify now using this link.",
        "Congratulations! You won 50000. Click this link to claim your reward.",
        "Your card is suspended. Send your OTP immediately to reactivate it.",
        "Security alert! Confirm your PIN and password at this link now.",
        "You have received a cash prize. Pay a small fee to release the money.",
        "Your credit card will be closed. Verify your details immediately.",
        "Unusual activity detected. Login through this link to secure your account.",
        "Claim your instant loan by sharing your OTP and card details.",
    ]
    rows = []
    for _ in range(150):
        rows.append((legitimate[_ % len(legitimate)], 0))
        rows.append((fraud[_ % len(fraud)], 1))
    return pd.DataFrame(rows, columns=["message", "fraud"])

def train_models():
    credit = _make_credit_data()
    fraud = _make_fraud_data()
    messages = _make_messages()

    credit.to_csv(DATA_DIR / "credit_card_data.csv", index=False)
    fraud.to_csv(DATA_DIR / "transactions.csv", index=False)
    messages.to_csv(DATA_DIR / "messages.csv", index=False)

    Xc = credit[CREDIT_FEATURES]
    yc = credit["default"]
    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
    credit_model = RandomForestClassifier(
        n_estimators=250, max_depth=9, min_samples_leaf=3, random_state=42, class_weight="balanced"
    )
    credit_model.fit(Xtr, ytr)
    cp = credit_model.predict(Xte)
    cprob = credit_model.predict_proba(Xte)[:, 1]
    cm = _metrics(yte, cp, cprob)
    joblib.dump(credit_model, MODEL_DIR / "credit_model.joblib")

    fraud_features = ["amount", "hour", "distance_km", "frequency_24h", "new_device", "international", "merchant_risk"]
    Xf = fraud[fraud_features]
    yf = fraud["fraud"]
    Xtr, Xte, ytr, yte = train_test_split(Xf, yf, test_size=0.2, random_state=42, stratify=yf)
    fraud_model = RandomForestClassifier(
        n_estimators=250, max_depth=10, min_samples_leaf=2, random_state=42, class_weight="balanced"
    )
    fraud_model.fit(Xtr, ytr)
    fp = fraud_model.predict(Xte)
    fprob = fraud_model.predict_proba(Xte)[:, 1]
    fm = _metrics(yte, fp, fprob)
    joblib.dump(fraud_model, MODEL_DIR / "fraud_model.joblib")

    Xm = messages["message"]
    ym = messages["fraud"]
    Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=0.2, random_state=42, stratify=ym)
    message_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    message_model.fit(Xtr, ytr)
    mp = message_model.predict(Xte)
    mprob = message_model.predict_proba(Xte)[:, 1]
    mm = _metrics(yte, mp, mprob)
    joblib.dump(message_model, MODEL_DIR / "message_model.joblib")

    metrics = {
        "credit": cm, "fraud": fm, "message": mm,
        "training_rows": {"credit": len(credit), "fraud": len(fraud), "message": len(messages)},
        "version": "1.0.0"
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

def ensure_models():
    needed = [
        MODEL_DIR / "credit_model.joblib",
        MODEL_DIR / "fraud_model.joblib",
        MODEL_DIR / "message_model.joblib",
        MODEL_DIR / "metrics.json",
    ]
    if not all(p.exists() for p in needed):
        train_models()

def load_credit():
    ensure_models()
    return joblib.load(MODEL_DIR / "credit_model.joblib")

def load_fraud():
    ensure_models()
    return joblib.load(MODEL_DIR / "fraud_model.joblib")

def load_message():
    ensure_models()
    return joblib.load(MODEL_DIR / "message_model.joblib")

def metrics():
    ensure_models()
    return json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))

def risk_label(prob):
    if prob < 0.30:
        return "LOW"
    if prob < 0.70:
        return "MEDIUM"
    return "HIGH"
