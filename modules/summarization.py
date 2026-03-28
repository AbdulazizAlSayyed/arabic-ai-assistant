"""
Text Summarization Module
Uses HuggingFace mT5 model (free) with fallback extractive summarization
"""

import os
import re
from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_summarizer():
    try:
        hf_token = os.environ.get("HF_TOKEN")
        summarizer = pipeline(
            "summarization",
            model="csebuetnlp/mT5_multilingual_XLSum",
            device=-1,
            token=hf_token
        )
        return summarizer
    except Exception as e:
        print(f"Error loading summarizer: {e}")
        return None

def summarize_text(text: str, max_length: int = 150) -> dict:
    original_length = len(text.split())
    if not text or original_length < 20:
        return {
            "summary": text if text else "النص قصير جداً للتلخيص",
            "method": "نص قصير (لا حاجة للتلخيص)",
            "original_length": original_length,
            "summary_length": len(text.split()) if text else 0
        }
    summarizer = load_summarizer()
    if summarizer:
        try:
            text_to_summarize = text[:1024]
            result = summarizer(
                text_to_summarize,
                max_length=max_length,
                min_length=30,
                do_sample=False,
                truncation=True
            )
            summary = result[0]['summary_text']
            summary_length = len(summary.split())
            return {
                "summary": summary,
                "method": "HuggingFace mT5 (AI Model - Free)",
                "original_length": original_length,
                "summary_length": summary_length
            }
        except Exception as e:
            print(f"Summarization error: {e}")
    # Fallback: extract first 3 sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    summary = ". ".join(sentences[:3])
    if len(sentences) > 3:
        summary += "..."
    summary_length = len(summary.split())
    return {
        "summary": summary,
        "method": "استخراج الجمل الأولى (بديل - مجاني)",
        "original_length": original_length,
        "summary_length": summary_length
    }