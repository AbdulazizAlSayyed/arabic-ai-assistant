"""
Machine Translation Module
Arabic ↔ English translation using MarianMT models
"""

import re
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

@st.cache_resource
def load_translation_models():
    """Load translation models with caching"""
    model_name_ar_en = "Helsinki-NLP/opus-mt-ar-en"
    tokenizer_ar_en = MarianTokenizer.from_pretrained(model_name_ar_en)
    model_ar_en = MarianMTModel.from_pretrained(model_name_ar_en)
    
    model_name_en_ar = "Helsinki-NLP/opus-mt-en-ar"
    tokenizer_en_ar = MarianTokenizer.from_pretrained(model_name_en_ar)
    model_en_ar = MarianMTModel.from_pretrained(model_name_en_ar)
    
    return tokenizer_ar_en, model_ar_en, tokenizer_en_ar, model_en_ar

_tokenizer_ar_en, _model_ar_en, _tokenizer_en_ar, _model_en_ar = load_translation_models()

def detect_language(text: str) -> str:
    arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
    return "ar" if arabic_chars else "en"

def translate_text(text: str, source_lang: str = "ar", target_lang: str = "en") -> str:
    if not text or not text.strip():
        return ""
    try:
        if source_lang == "ar" and target_lang == "en":
            batch = _tokenizer_ar_en([text], return_tensors="pt", padding=True)
            generated = _model_ar_en.generate(**batch)
            return _tokenizer_ar_en.decode(generated[0], skip_special_tokens=True)
        elif source_lang == "en" and target_lang == "ar":
            text = ">>ara<< " + text
            batch = _tokenizer_en_ar([text], return_tensors="pt", padding=True)
            generated = _model_en_ar.generate(**batch)
            return _tokenizer_en_ar.decode(generated[0], skip_special_tokens=True)
        else:
            return text
    except Exception as e:
        return f"Translation error: {str(e)}"

def batch_translate(texts: list, source_lang: str = "ar", target_lang: str = "en") -> list:
    results = []
    for text in texts:
        results.append(translate_text(text, source_lang, target_lang))
    return results