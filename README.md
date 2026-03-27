# 🤖 Arabic Virtual AI Assistant

A comprehensive Arabic AI assistant with multi-task NLP capabilities combining traditional Arabic NLP techniques with modern LLM-based approaches.

## ✨ Features

- 🌍 **Machine Translation**: Arabic ↔ English translation using MarianMT models
- 😊 **Sentiment Analysis**: ML-based + rule-based sentiment detection (Arabic & English)
- 🗣️ **Dialect Identification**: Traditional NLP pattern matching for Arabic dialects
- 📝 **Text Summarization**: LLM-based summarization of Arabic texts
- 🧠 **RAG Question Answering**: Retrieval-augmented generation with knowledge base

## 🛠️ Technologies

- **Traditional NLP**: Rule-based pattern matching, TF-IDF, Logistic Regression
- **LLM**: OpenAI GPT-4o-mini, Transformers
- **Embeddings**: Sentence Transformers (multilingual-MiniLM)
- **UI**: Streamlit

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/arabic-ai-assistant.git
cd arabic-ai-assistant

# Create virtual environment (Python 3.10+)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"  # On Windows: setx OPENAI_API_KEY "your-key"