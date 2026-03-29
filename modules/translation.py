from transformers import MarianMTModel, MarianTokenizer
import streamlit as st
import re

@st.cache_resource
def load_translation_models():
    """Load translation models with caching for performance"""
    # Arabic to English
    model_name_ar_en = "Helsinki-NLP/opus-mt-ar-en"
    tokenizer_ar_en = MarianTokenizer.from_pretrained(model_name_ar_en)
    model_ar_en = MarianMTModel.from_pretrained(model_name_ar_en)
    
    # English to Arabic
    model_name_en_ar = "Helsinki-NLP/opus-mt-en-ar"
    tokenizer_en_ar = MarianTokenizer.from_pretrained(model_name_en_ar)
    model_en_ar = MarianMTModel.from_pretrained(model_name_en_ar)
    
    return tokenizer_ar_en, model_ar_en, tokenizer_en_ar, model_en_ar

# Load models once
_tokenizer_ar_en, _model_ar_en, _tokenizer_en_ar, _model_en_ar = load_translation_models()

def detect_language(text: str) -> str:
    """Detect if text is Arabic or English"""
    arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
    return "ar" if arabic_chars else "en"

def translate_text(text: str, source_lang: str = "ar", target_lang: str = "en") -> str:
    """
    Translate text between Arabic and English
    
    Args:
        text: Text to translate
        source_lang: Source language code ('ar' or 'en')
        target_lang: Target language code ('ar' or 'en')
    
    Returns:
        Translated text
    """
    if not text or not text.strip():
        return ""
    
    try:
        if source_lang == "ar" and target_lang == "en":
            # Arabic to English
            batch = _tokenizer_ar_en([text], return_tensors="pt", padding=True)
            generated = _model_ar_en.generate(**batch)
            return _tokenizer_ar_en.decode(generated[0], skip_special_tokens=True)
        
        elif source_lang == "en" and target_lang == "ar":
            # English to Arabic (add prefix for better results)
            text = ">>ara<< " + text
            batch = _tokenizer_en_ar([text], return_tensors="pt", padding=True)
            generated = _model_en_ar.generate(**batch)
            return _tokenizer_en_ar.decode(generated[0], skip_special_tokens=True)
        
        else:
            return text
            
    except Exception as e:
        return f"Translation error: {str(e)}"