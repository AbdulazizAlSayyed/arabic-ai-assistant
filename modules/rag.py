"""
Simple LLM Question Answering (OpenAI)
No Retrieval - Direct AI Answer
"""

import os
from openai import OpenAI


# ==============================
# Load OpenAI client
# ==============================
def get_client():
    #api_key = "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"
    

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY not found in environment variables")

    return OpenAI(api_key=api_key)


# ==============================
# Main function
# ==============================
def answer_question(question: str) -> dict:
    client = get_client()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # سريع ورخيص
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. "
                        "Answer clearly and concisely. "
                        "Support both Arabic and English."
                    ),
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            temperature=0.5,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "method": "OpenAI LLM",
            "retrieved_chunks": []  # خليها فاضية حتى ما يكسر ال UI
        }

    except Exception as e:
        return {
            "answer": f"❌ Error: {str(e)}",
            "method": "Error",
            "retrieved_chunks": []
        }
