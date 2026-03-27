import re
import streamlit as st

# This is your TRADITIONAL ARABIC NLP COMPONENT
# Using rule-based patterns for dialect identification

@st.cache_resource
def load_dialect_patterns():
    """Load dialect patterns (Traditional NLP approach)"""
    
    # Dialect patterns based on common linguistic features
    patterns = {
        "Egyptian": {
            "keywords": ["إيه", "ليه", "أه", "أيوه", "بتاع", "أوي", "كده", "إزاي"],
            "prefixes": ["ه", "ب", "أ", "ن"],
            "suffixes": ["ت", "ش", "ك"],
            "unique": ["جدا", "قوي", "أوي", "أه", "أيوة"]
        },
        "Levantine": {
            "keywords": ["شو", "إيش", "أيش", "عم", "بدي", "أنا", "إنت", "هيدا", "هاد"],
            "prefixes": ["ب", "ع", "م", "ت"],
            "suffixes": ["ي", "ك", "ها", "نا"],
            "unique": ["شو", "عم", "بدي", "هيدا", "هاد", "هيدي"]
        },
        "Gulf": {
            "keywords": ["شنو", "وش", "إش", "مال", "حال", "قعد", "شلون", "أبشر"],
            "prefixes": ["أ", "ي", "ت", "ن"],
            "suffixes": ["ون", "ين", "وا", "ن"],
            "unique": ["شنو", "وش", "مالك", "شلون", "أبشر", "الحين"]
        },
        "Maghrebi": {
            "keywords": ["واش", "شنو", "دابا", "بزاف", "هذا", "هاذي", "فين", "كيفاش"],
            "prefixes": ["ك", "ت", "ن", "ي"],
            "suffixes": ["ش", "ت", "ك", "نا"],
            "unique": ["واش", "دابا", "بزاف", "كيفاش", "فين", "هاذي"]
        }
    }
    
    # MSA markers (Modern Standard Arabic)
    msa_markers = {
        "keywords": ["إن", "أن", "قد", "سوف", "لن", "إنما", "بينما", "عندما", "التي", "الذي"],
        "prefixes": ["س", "سوف", "ل", "ي"],
        "suffixes": ["ون", "ين", "ات", "ان"]
    }
    
    return patterns, msa_markers

def identify_dialect(text: str) -> dict:
    """
    Identify Arabic dialect using rule-based traditional NLP approach
    
    Returns:
        dict: {"dialect": str, "method": str, "confidence": float, "features": dict}
    """
    patterns, msa_markers = load_dialect_patterns()
    
    # Clean text
    text = text.lower()
    
    # Score each dialect
    scores = {"MSA": 0, "Egyptian": 0, "Levantine": 0, "Gulf": 0, "Maghrebi": 0}
    feature_matches = {}
    
    # Check for MSA markers
    msa_score = 0
    for keyword in msa_markers["keywords"]:
        if keyword in text:
            msa_score += 2
            feature_matches[f"msa_{keyword}"] = True
    
    scores["MSA"] = msa_score
    
    # Check each dialect
    for dialect, pattern in patterns.items():
        score = 0
        matches = []
        
        # Check keywords
        for keyword in pattern["keywords"]:
            if keyword in text:
                score += 3
                matches.append(keyword)
        
        # Check prefixes
        for prefix in pattern["prefixes"]:
            if any(word.startswith(prefix) for word in text.split()):
                score += 1
                matches.append(f"prefix_{prefix}")
        
        # Check suffixes
        for suffix in pattern["suffixes"]:
            if any(word.endswith(suffix) for word in text.split()):
                score += 1
                matches.append(f"suffix_{suffix}")
        
        # Check unique markers
        for unique in pattern["unique"]:
            if unique in text:
                score += 4
                matches.append(unique)
        
        scores[dialect] = score
        feature_matches[dialect] = matches
    
    # Determine dialect
    max_score = max(scores.values())
    
    if max_score == 0:
        # Fallback: detect based on common words
        if any(marker in text for marker in ["أنا", "نحن", "هو", "هي"]):
            dialect = "MSA"
            confidence = 0.5
        else:
            dialect = "Unknown"
            confidence = 0.0
    else:
        # Get dialect with highest score
        dialect = max(scores, key=scores.get)
        confidence = min(scores[dialect] / (max_score + 10), 0.95)
    
    return {
        "dialect": dialect,
        "method": "Rule-based (Traditional NLP - Pattern Matching)",
        "confidence": confidence,
        "scores": scores,
        "features": feature_matches
    }