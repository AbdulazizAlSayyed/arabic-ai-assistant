import os
import re
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.joblib")
DATA_PATH = "data/sentiment/sentiment.csv"

_sentiment_model = None


# ==============================
# Normalize Arabic
# ==============================
def normalize_arabic(text: str) -> str:
    text = str(text).strip().lower()

    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==============================
# Train Model (FIXED 🔥)
# ==============================
def train_sentiment_model(csv_path: str = DATA_PATH):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # 🔥 normalize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # 🔥 auto rename columns
    rename_map = {
        "polarity": "label",
        "sentiment": "label",
        "class": "label",
        "target": "label",
        "text": "text",
        "review": "text",
        "sentence": "text"
    }
    df.rename(columns=rename_map, inplace=True)

    # ✅ final check
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Columns found: {df.columns.tolist()}")

    # clean data
    df = df.dropna(subset=["text", "label"]).copy()

    df["text"] = df["text"].astype(str).apply(normalize_arabic)
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    # model
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(df["text"], df["label"])

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model


# ==============================
# Load Model
# ==============================
def load_sentiment_model():
    global _sentiment_model

    if _sentiment_model is not None:
        return _sentiment_model

    if os.path.exists(MODEL_PATH):
        _sentiment_model = joblib.load(MODEL_PATH)
        return _sentiment_model

    if os.path.exists(DATA_PATH):
        _sentiment_model = train_sentiment_model(DATA_PATH)
        return _sentiment_model

    return None


# ==============================
# Rule-based fallback
# ==============================
POSITIVE_WORDS = {
    "ممتاز", "رائع", "جميل", "ممتع", "جيد", "حلو", "احب", "سريع", "مفيد","ممتازه", "روعة", "حلوه", "عظيم", "كويس"
}

NEGATIVE_WORDS = {
    "سيء", "سيئ", "رديء", "بطيء", "مشكله", "مشكلة", "غالي", "زفت", "تعبان","سيئه", "زبالة", "خايس", "ضعيف"
}


def rule_based_sentiment(text: str) -> str:
    text_n = normalize_arabic(text)
    tokens = set(text_n.split())

    pos_score = len(tokens & POSITIVE_WORDS)
    neg_score = len(tokens & NEGATIVE_WORDS)

    if pos_score > neg_score:
        return "positive"
    elif neg_score > pos_score:
        return "negative"
    return "neutral"


# ==============================
# Main API
# ==============================

def analyze_sentiment(text: str) -> dict:
    if not text or not text.strip():
        return {
            "label": "neutral",
            "score": 0.0,
            "method": "rule_based"
        }

    clean_text = normalize_arabic(text)
    tokens = clean_text.split()

    # ==========================
    # 🔥 RULE PRIORITY (short text)
    # ==========================
    if len(tokens) <= 2:
        rule_pred = rule_based_sentiment(clean_text)
        return {
            "label": rule_pred,
            "score": 0.9,
            "method": "rule_based"
        }

    model = load_sentiment_model()

    # ==========================
    # ML model
    # ==========================
    if model is not None:
        pred = model.predict([clean_text])[0]

        try:
            proba = model.predict_proba([clean_text])[0]
            score = float(max(proba))
        except:
            score = 0.7

        return {
            "label": pred,
            "score": round(score, 3),
            "method": "ml"
        }

    # fallback
    pred = rule_based_sentiment(clean_text)

    return {
        "label": pred,
        "score": 0.5,
        "method": "rule_based"
    }