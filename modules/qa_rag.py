import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

KB_DIR = "data/kb"
_embedder = None
_chunks = []
_chunk_embeddings = None

def _load_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _embedder

def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Please set it as an environment variable."
        )
    return OpenAI(api_key=api_key)

def load_kb_documents(kb_dir: str = KB_DIR) -> List[Dict]:
    docs = []
    if not os.path.exists(kb_dir):
        return docs
    for filename in os.listdir(kb_dir):
        if filename.endswith(".txt"):
            path = os.path.join(kb_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    docs.append({
                        "source": filename,
                        "text": text
                    })
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
            all_chunks.append({
                "source": doc["source"],
                "text": c
            })
    _chunks = all_chunks
    if not _chunks:
        _chunk_embeddings = None
        return
    embedder = _load_embedder()
    _chunk_embeddings = embedder.encode(
        [c["text"] for c in _chunks],
        convert_to_tensor=True
    )

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
        results.append({
            "score": float(score),
            "source": _chunks[idx]["source"],
            "text": _chunks[idx]["text"]
        })
    return results

def build_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    context = "\n\n".join(
        [f"المصدر: {c['source']}\n{c['text']}" for c in retrieved_chunks]
    )

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

def fallback_answer(retrieved_chunks: List[Dict]) -> str:
    if not retrieved_chunks:
        return "لم أجد معلومات كافية في قاعدة المعرفة للإجابة على هذا السؤال."
    return (
        "بناءً على المعلومات المسترجعة من قاعدة المعرفة، هذه هي الإجابة الأقرب:\n\n"
        f"{retrieved_chunks[0]['text']}"
    )

def generate_answer(question: str, retrieved_chunks: List[Dict]) -> str:
    if not retrieved_chunks:
        return "لم أجد معلومات كافية في قاعدة المعرفة للإجابة على هذا السؤال."
    prompt = build_prompt(question, retrieved_chunks)
    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "أنت مساعد عربي ذكي يجيب بدقة وباختصار اعتماداً على السياق فقط."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return fallback_answer(retrieved_chunks)

def answer_question(question: str, top_k: int = 3) -> dict:
    if not question or not question.strip():
        return {
            "question": question,
            "answer": "الرجاء إدخال سؤال.",
            "retrieved_chunks": []
        }
    retrieved = retrieve_context(question, top_k=top_k)
    answer = generate_answer(question, retrieved)
    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved
    }
