"""
Smart Summarization Module
Supports Arabic + English using translation pipeline
"""

import re
import streamlit as st
from transformers import pipeline
from modules.translation import translate_text


# ==============================
# Load summarizer (lightweight)
# ==============================
@st.cache_resource
def load_summarizer():
    try:
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=-1
        )
        return summarizer
    except Exception as e:
        print(f"Error loading summarizer: {e}")
        return None


# ==============================
# Detect Arabic
# ==============================
def is_arabic(text):
    return any("\u0600" <= c <= "\u06FF" for c in text)


# ==============================
# Main function
# ==============================
def summarize_text(text: str, max_length: int = 150) -> dict:
    original_length = len(text.split())

    if not text or original_length < 20:
        return {
            "summary": text if text else "النص قصير جداً للتلخيص",
            "method": "Short Text",
            "original_length": original_length,
            "summary_length": len(text.split()) if text else 0
        }

    summarizer = load_summarizer()

    try:
        # ==========================
        # 🇬🇧 English
        # ==========================
        if not is_arabic(text) and summarizer:
            result = summarizer(
                text[:1024],
                max_length=max_length,
                min_length=30,
                do_sample=False,
                truncation=True
            )

            summary = result[0]['summary_text']

            return {
                "summary": summary,
                "method": "English AI (DistilBART)",
                "original_length": original_length,
                "summary_length": len(summary.split())
            }

        # ==========================
        # 🇸🇦 Arabic (translate pipeline)
        # ==========================
        if is_arabic(text) and summarizer:
            # 1. AR → EN
            translated = translate_text(text, target_lang="en")

            # 2. Summarize
            result = summarizer(
                translated[:1024],
                max_length=max_length,
                min_length=30,
                do_sample=False,
                truncation=True
            )

            summary_en = result[0]['summary_text']

            # 3. EN → AR
            summary_ar = translate_text(summary_en, target_lang="ar")

            return {
                "summary": summary_ar,
                "method": "Arabic AI (Translate + DistilBART)",
                "original_length": original_length,
                "summary_length": len(summary_ar.split())
            }

    except Exception as e:
        print(f"Summarization error: {e}")

    # ==========================
    # 🔥 Fallback (any language)
    # ==========================
    sentences = re.split(r'[.!?؟]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    summary = ". ".join(sentences[:3])
    if len(sentences) > 3:
        summary += "..."

    return {
        "summary": summary,
        "method": "Extractive Fallback",
        "original_length": original_length,
        "summary_length": len(summary.split())
    }