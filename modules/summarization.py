import os
from openai import OpenAI

def _get_openai_client():
    """Get OpenAI client with API key from environment"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)

def summarize_text(text: str, max_length: int = 150) -> dict:
    """
    Summarize text using LLM
    
    Returns:
        dict: {"summary": str, "method": str}
    """
    if not text or len(text.split()) < 20:
        return {
            "summary": text if text else "Text too short for summarization",
            "method": "Fallback (text too short)"
        }
    
    try:
        client = _get_openai_client()
        
        prompt = f"""
قم بتلخيص النص التالي إلى ملخص قصير ومختصر. 
يجب أن يكون الملخص:
- باللغة العربية الفصحى
- لا يتجاوز {max_length} كلمة
- يحتفظ بالنقاط الرئيسية فقط

النص:
{text}

الملخص:
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "أنت مساعد متخصص في تلخيص النصوص العربية. ملخصاتك دقيقة ومختصرة."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        summary = response.choices[0].message.content.strip()
        
        return {
            "summary": summary,
            "method": "LLM (GPT-4o-mini)"
        }
        
    except Exception as e:
        # Fallback: extract first few sentences
        sentences = text.split(".")[:3]
        summary = ". ".join(sentences)[:300]
        
        return {
            "summary": summary + "...",
            "method": f"Fallback (extractive - {str(e)[:50]})"
        }