from transformers import MarianMTModel, MarianTokenizer

# load models once
model_name_ar_en = "Helsinki-NLP/opus-mt-ar-en"
tokenizer_ar_en = MarianTokenizer.from_pretrained(model_name_ar_en)
model_ar_en = MarianMTModel.from_pretrained(model_name_ar_en)

model_name_en_ar = "Helsinki-NLP/opus-mt-en-ar"
tokenizer_en_ar = MarianTokenizer.from_pretrained(model_name_en_ar)
model_en_ar = MarianMTModel.from_pretrained(model_name_en_ar)


def translate_text(text, source_lang="ar", target_lang="en"):

    if source_lang == "ar" and target_lang == "en":
        batch = tokenizer_ar_en([text], return_tensors="pt", padding=True)
        generated = model_ar_en.generate(**batch)
        return tokenizer_ar_en.decode(generated[0], skip_special_tokens=True)

    if source_lang == "en" and target_lang == "ar":
        text = ">>ara<< " + text
        batch = tokenizer_en_ar([text], return_tensors="pt", padding=True)
        generated = model_en_ar.generate(**batch)
        return tokenizer_en_ar.decode(generated[0], skip_special_tokens=True)

    return text
