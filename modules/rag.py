"""
🔥 Advanced RAG Module (Arabic + English + Offline Safe)
"""

import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

KB_DIR = "data/kb"

_embedder = None
_chunks = []
_chunk_embeddings = None


# ==============================
# Load embedding model safely
# ==============================
def _load_embedder():
    global _embedder
    if _embedder is None:
        try:
            _embedder = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            print("✅ Embedding model loaded")
        except Exception as e:
            print("❌ Embedding failed:", e)
            _embedder = None
    return _embedder


# ==============================
# Load documents
# ==============================
def load_kb_documents(kb_dir: str = KB_DIR) -> List[Dict]:
    docs = []

    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir, exist_ok=True)

    for filename in os.listdir(kb_dir):
        if filename.endswith(".txt"):
            path = os.path.join(kb_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        docs.append({
                            "source": filename,
                            "text": text
                        })
            except Exception as e:
                print("Error loading:", filename, e)

    return docs


# ==============================
# Chunk text
# ==============================
def chunk_text(text: str, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks


# ==============================
# Build index
# ==============================
def build_index():
    global _chunks, _chunk_embeddings

    docs = load_kb_documents()

    _chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for c in chunks:
            _chunks.append({
                "source": doc["source"],
                "text": c
            })

    embedder = _load_embedder()

    if embedder is None or not _chunks:
        _chunk_embeddings = None
        return

    _chunk_embeddings = embedder.encode(
        [c["text"] for c in _chunks],
        convert_to_tensor=True
    )


# ==============================
# Retrieve
# ==============================
def retrieve_context(question: str, top_k=3):
    global _chunks, _chunk_embeddings

    if not _chunks:
        build_index()

    embedder = _load_embedder()

    if embedder is None or _chunk_embeddings is None:
        print("⚠️ Using fallback retrieval")
        return _chunks[:top_k]  # simple fallback

    q_emb = embedder.encode(question, convert_to_tensor=True)

    scores = util.cos_sim(q_emb, _chunk_embeddings)[0]
    top = scores.topk(k=min(top_k, len(_chunks)))

    results = []
    for score, idx in zip(top.values, top.indices):
        idx = int(idx)
        results.append({
            "score": float(score),
            "source": _chunks[idx]["source"],
            "text": _chunks[idx]["text"]
        })

    return results


# ==============================
# Generate answer (SMART)
# ==============================
def generate_answer(question: str, chunks: List[Dict]) -> str:
    if not chunks:
        return "❌ No relevant information found."

    best = chunks[0]

    return f"""
📌 Answer:

{best['text']}

📄 Source: {best['source']}
🎯 Confidence: {best.get('score', 0):.2%}
"""


# ==============================
# MAIN FUNCTION
# ==============================
def answer_question(question: str, top_k=3) -> dict:
    if not question.strip():
        return {
            "answer": "⚠️ Please enter a question",
            "method": "No Input"
        }

    chunks = retrieve_context(question, top_k)

    print("DEBUG chunks:", chunks)

    answer = generate_answer(question, chunks)

    return {
        "question": question,
        "answer": answer,
        "chunks": chunks,
        "method": "RAG (Multilingual + Offline Safe)"
    }


# ==============================
# Check KB
# ==============================
def check_knowledge_base():
    docs = load_kb_documents()

    if not docs:
        return False, "⚠️ No documents found in data/kb/"

    return True, f"✅ Loaded {len(docs)} documents"