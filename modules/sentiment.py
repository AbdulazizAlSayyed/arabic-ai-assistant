#import libraries
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
POSITIVE_WORDS = {
    "ممتاز", "رائع", "جميل", "ممتعه", "ممتع", "جيد", "حلو", "احب", "سريع", "مفيد"
}
NEGATIVE_WORDS = {
    "سيء", "سيئ", "رديء", "بطيء", "مشكله", "مشكلة", "غالي", "زفت", "كريه", "تعبان"
}

def normalize_arabic(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)  # remove diacritics / tatweel
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_sentiment_model(csv_path: str = DATA_PATH):
    """
    Expected CSV columns:
    - text
    - label   (positive / negative / neutral)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Sentiment dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).apply(normalize_arabic)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    model.fit(df["text"], df["label"])
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model

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

def rule_based_sentiment(text: str) -> str:
    text_n = normalize_arabic(text)
    tokens = set(text_n.split())
    pos_score = len(tokens & POSITIVE_WORDS)
    neg_score = len(tokens & NEGATIVE_WORDS)
    if pos_score > neg_score:
        return "positive"
    if neg_score > pos_score:
        return "negative"
    return "neutral"

def analyze_sentiment(text: str) -> dict:
    """
    Returns:
        {
            "label": "positive" / "negative" / "neutral",
            "method": "ml" or "rule_based"
        }
    """
    if not text or not text.strip():
        return {"label": "neutral", "method": "rule_based"}
    model = load_sentiment_model()
    clean_text = normalize_arabic(text)
    if model is not None:
        pred = model.predict([clean_text])[0]
        return {"label": pred, "method": "ml"}

    pred = rule_based_sentiment(clean_text)
    return {"label": pred, "method": "rule_based"}
