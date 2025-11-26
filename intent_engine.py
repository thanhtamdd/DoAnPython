# intent_engine.py
import re
import unidecode
import joblib
import pandas as pd
from pathlib import Path

# ---------- utils ----------
def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = unidecode.unidecode(text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

# ---------- load model ----------
MODEL_PATH = Path("intent_model.pkl")
VECT_PATH = Path("vectorizer.pkl")
RESP_AUG_PATH = Path("responses_augmented.csv")

_model = None
_vectorizer = None
RESPONSES = {}

def _load_model_and_data():
    global _model, _vectorizer, RESPONSES
    if _model is None or _vectorizer is None:
        if MODEL_PATH.exists() and VECT_PATH.exists():
            _model = joblib.load(str(MODEL_PATH))
            _vectorizer = joblib.load(str(VECT_PATH))
        else:
            raise FileNotFoundError("Missing intent_model.pkl or vectorizer.pkl. Run train_model.py first.")
    if RESP_AUG_PATH.exists():
        df = pd.read_csv(str(RESP_AUG_PATH))
        if "intent" in df.columns and "response" in df.columns:
            RESPONSES = dict(zip(df["intent"], df["response"]))
        else:
            RESPONSES = {}
    else:
        RESPONSES = {}

# call on import
_load_model_and_data()

# ---------- predict ----------
def predict_intent(text: str, threshold: float = 0.55) -> str:
    """
    Return predicted intent (string). If confidence < threshold -> 'unknown'
    """
    text_clean = clean_text(text)
    X = _vectorizer.transform([text_clean])
    probs = _model.predict_proba(X)[0]
    pred = _model.classes_[probs.argmax()]
    if probs.max() < threshold:
        return "unknown"
    return pred

# ---------- helper to get response from RESPONSES ----------
def get_response_for_intent(intent: str) -> str:
    return RESPONSES.get(intent, None)
