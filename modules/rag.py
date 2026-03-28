"""
RAG Question Answering Module
Tries OpenAI first, falls back to retrieval-only
"""

import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

KB_DIR = "data/kb"
_embedder = None
_chunks = []
_chunk_embeddings = None

def _load_embedder():
    global _embedder
    if _embedder is None:
        hf_token = os.environ.get("HF_TOKEN")
        _embedder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            token=hf_token
        )
    return _embedder

def load_kb_documents(kb_dir: str = KB_DIR) -> List[Dict]:
    docs = []
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir, exist_ok=True)
        if not os.listdir(kb_dir):
            sample_path = os.path.join(kb_dir, "sample_info.txt")
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write("""
                لبنان هي دولة عربية تقع في غرب آسيا. عاصمتها بيروت.
                اللغة الرسمية هي العربية. العملة هي الليرة اللبنانية.
                يبلغ عدد السكان حوالي 5.5 مليون نسمة.
                لبنان عضو في جامعة الدول العربية والأمم المتحدة.
                """)
        return docs
    for filename in os.listdir(kb_dir):
        if filename.endswith(".txt"):
            path = os.path.join(kb_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        docs.append({"source": filename, "text": text})
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return docs

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be greater than overlap")
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks

def build_index():
    global _chunks, _chunk_embeddings
    docs = load_kb_documents()
    all_chunks = []
    for doc in docs:
        text_chunks = chunk_text(doc["text"])
        for c in text_chunks:
            all_chunks.append({"source": doc["source"], "text": c})
    _chunks = all_chunks
    if not _chunks:
        _chunk_embeddings = None
        return
    embedder = _load_embedder()
    _chunk_embeddings = embedder.encode([c["text"] for c in _chunks], convert_to_tensor=True)

def retrieve_context(question: str, top_k: int = 3) -> List[Dict]:
    global _chunks, _chunk_embeddings
    if not _chunks or _chunk_embeddings is None:
        build_index()
    if not _chunks or _chunk_embeddings is None:
        return []
    embedder = _load_embedder()
    q_emb = embedder.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, _chunk_embeddings)[0]
    top_results = scores.topk(k=min(top_k, len(_chunks)))
    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        idx = int(idx)
        results.append({"score": float(score), "source": _chunks[idx]["source"], "text": _chunks[idx]["text"]})
    return results

def build_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    context = "\n\n".join([f"المصدر: {c['source']}\n{c['text']}" for c in retrieved_chunks[:3]])
    prompt = f"""
أجب فقط باستخدام المعلومات الموجودة في السياق التالي.
إذا لم تجد الإجابة بوضوح في السياق، قل:
"لم أجد معلومات كافية في قاعدة المعرفة."

السياق:
{context}

السؤال:
{question}

أجب بالعربية بشكل واضح ومختصر، ولا تضف معلومات من خارج السياق.
"""
    return prompt.strip()

def generate_answer_with_openai(question: str, retrieved_chunks: List[Dict]) -> str:
    if not retrieved_chunks:
        return "لم أجد معلومات كافية في قاعدة المعرفة للإجابة على هذا السؤال."
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        client = OpenAI(api_key=api_key)
        prompt = build_prompt(question, retrieved_chunks)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "أنت مساعد عربي ذكي يجيب بدقة وباختصار اعتماداً على السياق فقط."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None

def generate_answer_fallback(question: str, retrieved_chunks: List[Dict]) -> str:
    if not retrieved_chunks:
        return "لم أجد معلومات كافية في قاعدة المعرفة للإجابة على هذا السؤال."
    best_chunk = retrieved_chunks[0]
    return f"""
📌 **الإجابة المستندة إلى قاعدة المعرفة:**

{best_chunk['text']}

📚 **المصدر:** {best_chunk['source']}
🎯 **درجة الصلة:** {best_chunk['score']:.2%}
"""

def answer_question(question: str, top_k: int = 3) -> dict:
    if not question or not question.strip():
        return {"question": question, "answer": "الرجاء إدخال سؤال.", "retrieved_chunks": [], "method": "No input"}
    retrieved = retrieve_context(question, top_k=top_k)
    openai_answer = generate_answer_with_openai(question, retrieved)
    if openai_answer:
        return {
            "question": question,
            "answer": openai_answer,
            "retrieved_chunks": retrieved,
            "method": "OpenAI GPT-4o-mini (Smart AI)"
        }
    fallback_answer = generate_answer_fallback(question, retrieved)
    return {
        "question": question,
        "answer": fallback_answer,
        "retrieved_chunks": retrieved,
        "method": "Retrieval-only (No API - Direct Context)"
    }

def check_knowledge_base():
    docs = load_kb_documents()
    if not docs:
        return False, "⚠️ لا توجد مستندات في قاعدة المعرفة. أضف ملفات .txt إلى مجلد data/kb/"
    return True, f"✅ تم العثور على {len(docs)} مستند(ات) في قاعدة المعرفة"