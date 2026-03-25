#import library
from transformers import pipeline

#create translators
translator_ar_en = pipeline("translation", model = "Helsinki-NLP/opus-mt-ar-en") 
translator_en_ar = pipeline("translation", model = "Helsinki-NLP/opus-mt-en-ar")

#translating functions
def translate_text(text, source_lang="ar", target_lang="en"):
  if source_lang == "ar" and target_lang == "en":
    result = translator_ar_en(text)
    return result[0]["translation_text"]
  if source_lang == "en" and target_lang == "ar":
    result = translator_en_ar(text)
    return result[0]["translation_text"]
    return text
