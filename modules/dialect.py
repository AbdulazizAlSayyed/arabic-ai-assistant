"""
Dialect Identification Module
Traditional Arabic NLP component - Rule-based pattern matching
CAMeL Tools integration for ML-based identification
"""

import re
import streamlit as st

@st.cache_resource
def load_dialect_patterns():
    patterns = {
        "Egyptian": {
            "keywords": ["إيه", "ليه", "أه", "أيوه", "بتاع", "أوي", "كده", "إزاي", "عامل", "بقة", "أهو"],
            "prefixes": ["ه", "ب", "أ", "ن"],
            "suffixes": ["ت", "ش", "ك"],
            "unique": ["جدا", "قوي", "أوي", "أه", "أيوة", "بقة"]
        },
        "Levantine": {
            "keywords": ["شو", "إيش", "أيش", "عم", "بدي", "أنا", "إنت", "هيدا", "هاد", "هيدي", "هيك"],
            "prefixes": ["ب", "ع", "م", "ت"],
            "suffixes": ["ي", "ك", "ها", "نا"],
            "unique": ["شو", "عم", "بدي", "هيدا", "هاد", "هيدي"]
        },
        "Gulf": {
            "keywords": ["شنو", "وش", "إش", "مال", "حال", "قعد", "شلون", "أبشر", "الحين", "دحين"],
            "prefixes": ["أ", "ي", "ت", "ن"],
            "suffixes": ["ون", "ين", "وا", "ن"],
            "unique": ["شنو", "وش", "مالك", "شلون", "أبشر", "الحين"]
        },
        "Maghrebi": {
            "keywords": ["واش", "شنو", "دابا", "بزاف", "هذا", "هاذي", "فين", "كيفاش", "هاد", "هدا"],
            "prefixes": ["ك", "ت", "ن", "ي"],
            "suffixes": ["ش", "ت", "ك", "نا"],
            "unique": ["واش", "دابا", "بزاف", "كيفاش", "فين", "هاذي"]
        }
    }
    msa_markers = {
        "keywords": ["إن", "أن", "قد", "سوف", "لن", "إنما", "بينما", "عندما", "التي", "الذي", "الذين"],
        "prefixes": ["س", "سوف", "ل", "ي"],
        "suffixes": ["ون", "ين", "ات", "ان"]
    }
    return patterns, msa_markers

import platform

def identify_dialect_with_camel(text: str) -> dict:
    # ❌ عطّل CAMeL على Windows
    if platform.system() == "Windows":
        return None

    try:
        from camel_tools.dialectid import DialectIdentifier

        identifier = DialectIdentifier()
        result = identifier.identify(text)

        dialect_map = {
            "EGY": "Egyptian",
            "LEV": "Levantine",
            "GLF": "Gulf",
            "NOR": "MSA",
            "MAG": "Maghrebi"
        }

        detected = result.get('dialect', 'Unknown')
        dialect_name = dialect_map.get(detected, detected)

        return {
            "dialect": dialect_name,
            "method": "CAMeL Tools (ML Model)",
            "confidence": result.get('confidence', 0.5),
            "raw_result": result
        }

    except Exception:
        return None  # ❌ بدون print (منع spam)
def identify_dialect(text: str) -> dict:
    try:
        camel_result = identify_dialect_with_camel(text)
        if camel_result and camel_result.get('confidence', 0) > 0.5:
            return camel_result
    except:
        pass
    patterns, msa_markers = load_dialect_patterns()
    text = text.lower()
    scores = {"MSA": 0, "Egyptian": 0, "Levantine": 0, "Gulf": 0, "Maghrebi": 0}
    feature_matches = {}
    msa_score = 0
    for keyword in msa_markers["keywords"]:
        if keyword in text:
            msa_score += 2
            feature_matches[f"msa_{keyword}"] = True
    scores["MSA"] = msa_score
    for dialect, pattern in patterns.items():
        score = 0
        matches = []
        for keyword in pattern["keywords"]:
            if keyword in text:
                score += 3
                matches.append(keyword)
        for prefix in pattern["prefixes"]:
            if any(word.startswith(prefix) for word in text.split()):
                score += 1
                matches.append(f"prefix_{prefix}")
        for suffix in pattern["suffixes"]:
            if any(word.endswith(suffix) for word in text.split()):
                score += 1
                matches.append(f"suffix_{suffix}")
        for unique in pattern["unique"]:
            if unique in text:
                score += 4
                matches.append(unique)
        scores[dialect] = score
        feature_matches[dialect] = matches
    max_score = max(scores.values())
    if max_score == 0:
        if any(marker in text for marker in ["أنا", "نحن", "هو", "هي"]):
            dialect = "MSA"
            confidence = 0.5
        else:
            dialect = "Unknown"
            confidence = 0.0
    else:
        dialect = max(scores, key=scores.get)
        confidence = min(scores[dialect] / (max_score + 10), 0.95)
    return {
        "dialect": dialect,
        "method": "Rule-based (Traditional NLP - Pattern Matching)",
        "confidence": confidence,
        "scores": scores,
        "features": feature_matches
    }