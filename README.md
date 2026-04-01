# 🤖 Arabic Virtual AI Assistant

A comprehensive Arabic AI assistant with multi-task NLP capabilities, combining traditional Arabic NLP techniques with modern LLM-based approaches.

---

## ✨ Features

* 🌍 **Machine Translation**
  Arabic ↔ English translation using MarianMT models

* 😊 **Sentiment Analysis**
  Hybrid approach combining ML-based and rule-based sentiment detection (Arabic & English)

* 🗣️ **Dialect Identification**
  Traditional NLP pattern matching to detect Arabic dialects

* 📝 **Text Summarization**
  LLM-based summarization for Arabic texts

* 🧠 **RAG Question Answering**
  Retrieval-Augmented Generation (RAG) system with a custom knowledge base

---

## 🛠️ Technologies

* **Traditional NLP**
  TF-IDF, Logistic Regression, Rule-based pattern matching

* **LLM & Transformers**
  OpenAI GPT-4o-mini, Hugging Face Transformers

* **Embeddings**
  Sentence Transformers (multilingual-MiniLM)

* **Frontend**
  Streamlit

---

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
```

---

## 🚀 Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 📊 System Architecture

The system integrates multiple NLP components:

* Preprocessing module (Arabic text normalization)
* Task-specific models:

  * Translation model (MarianMT)
  * Sentiment classifier (ML + rules)
  * Dialect detector (pattern-based)
  * Summarization (LLM)
* RAG pipeline:

  * Embedding generation
  * Vector retrieval
  * LLM response generation

---

## 🧪 Example Use Cases

* Translate Arabic text to English
* Detect sentiment of user reviews
* Identify dialect (Levantine, Gulf, etc.)
* Summarize long Arabic articles
* Ask questions over a knowledge base

---

## 📈 Future Improvements

* Fine-tune Arabic-specific transformer models (e.g., AraBERT)
* Improve dialect classification using deep learning
* Add speech-to-text and voice assistant support
* Expand knowledge base for RAG system

---

## 👨‍💻 Author

**Abdulaziz Al Sayyed**
Computer Science Student – Lebanese American University

---

## 📄 License

This project is open-source and available under the MIT License.
