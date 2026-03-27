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

# Arabic sentiment keywords
ARABIC_POSITIVE = {
    "ممتاز", "رائع", "جميل", "ممتعه", "ممتع", "جيد", "حلو", "احب", "سريع", "مفيد",
    "رائعة", "جميلة", "جيدة", "حلوة", "سريعة", "مفيدة", "ممتازة", "عظيم", "شكرا",
    "ناجح", "سعيد", "فرح", "باهر", "خيالي", "مذهل", "طيب", "لطيف", "مميز", "فريد",
    "أفضل", "مسرور", "مبسوط", "حسن", "قوي"
}

ARABIC_NEGATIVE = {
    "سيء", "سيئ", "رديء", "بطيء", "مشكله", "مشكلة", "غالي", "زفت", "كريه", "تعبان",
    "سيئة", "رديئة", "بطيئة", "غالية", "كريهة", "مزعج", "مؤلم", "فاشل", "خاسر",
    "ضعيف", "خطأ", "غلط", "عيب", "صعب", "متعب", "مضجر", "كئيب", "حزين", "غاضب",
    "محبط", "مستاء", "زعلان", "بشع", "قبيح", "كره", "أكره"
}

# English sentiment keywords
ENGLISH_POSITIVE = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", 
    "beautiful", "love", "like", "happy", "nice", "awesome", "perfect",
    "best", "superb", "brilliant", "outstanding", "wonderful", "excited",
    "glad", "pleased", "delighted", "enjoy", "enjoyed", "lovely", "cool"
}

ENGLISH_NEGATIVE = {
    "bad", "terrible", "awful", "horrible", "worst", "hate", "dislike",
    "sad", "angry", "frustrated", "annoying", "poor", "disappointed",
    "useless", "failure", "wrong", "problem", "issue", "upset",
    "mad", "annoyed", "frustrating", "disgusting", "terrible"
}

def detect_language(text: str) -> str:
    """Detect if text is Arabic or English"""
    arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
    return "ar" if arabic_chars else "en"

def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for better analysis"""
    text = str(text).strip().lower()
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_english(text: str) -> str:
    """Normalize English text"""
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def train_sentiment_model(csv_path: str = DATA_PATH):
    """Train sentiment model from CSV file"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Sentiment dataset not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    
    df = df.dropna(subset=["text", "label"]).copy()
    
    # Normalize based on language
    def normalize_text(text):
        lang = detect_language(text)
        if lang == "ar":
            return normalize_arabic(text)
        else:
            return normalize_english(text)
    
    df["text"] = df["text"].astype(str).apply(normalize_text)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    model.fit(df["text"], df["label"])
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model

def load_sentiment_model():
    """Load sentiment model from disk or train if not exists"""
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

def rule_based_sentiment(text: str) -> dict:
    """Fallback rule-based sentiment analysis for both languages"""
    lang = detect_language(text)
    
    if lang == "ar":
        normalized = normalize_arabic(text)
        tokens = set(normalized.split())
        pos_score = len(tokens & ARABIC_POSITIVE)
        neg_score = len(tokens & ARABIC_NEGATIVE)
        method = "Rule-based (Arabic keywords)"
    else:
        normalized = normalize_english(text)
        tokens = set(normalized.split())
        pos_score = len(tokens & ENGLISH_POSITIVE)
        neg_score = len(tokens & ENGLISH_NEGATIVE)
        method = "Rule-based (English keywords)"
    
    # Calculate confidence based on score difference
    total = pos_score + neg_score
    if total > 0:
        confidence = min(abs(pos_score - neg_score) / total, 0.95)
    else:
        confidence = 0.5
    
    if pos_score > neg_score:
        return {"label": "positive", "method": method, "confidence": confidence}
    if neg_score > pos_score:
        return {"label": "negative", "method": method, "confidence": confidence}
    
    return {"label": "neutral", "method": method, "confidence": 0.5}

def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of text (supports Arabic and English)
    
    Returns:
        dict: {"label": str, "method": str, "confidence": float}
    """
    if not text or not text.strip():
        return {"label": "neutral", "method": "Fallback", "confidence": 0.0}
    
    model = load_sentiment_model()
    lang = detect_language(text)
    
    # Normalize based on language
    if lang == "ar":
        clean_text = normalize_arabic(text)
    else:
        clean_text = normalize_english(text)
    
    # Try ML model first
    if model is not None:
        try:
            pred = model.predict([clean_text])[0]
            proba = model.predict_proba([clean_text])[0]
            confidence = float(max(proba))
            
            return {
                "label": pred,
                "method": "Machine Learning (TF-IDF + Logistic Regression)",
                "confidence": confidence
            }
        except Exception:
            pass
    
    # Fallback to rule-based
    return rule_based_sentiment(text)