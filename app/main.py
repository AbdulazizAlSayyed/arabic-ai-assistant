import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.translation import translate_text, detect_language
from modules.sentiment import analyze_sentiment
from modules.rag import answer_question
from modules.dialect import identify_dialect
from modules.summarization import summarize_text

# Page configuration
st.set_page_config(
    page_title="Arabic AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        color: white;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🤖 Arabic AI Assistant</h1><p>Multi-Task NLP Assistant with Traditional + LLM Approaches</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.markdown("## 🚀 Features")
    st.markdown("""
    - 🌍 **Machine Translation** (Arabic ↔ English)
    - 😊 **Sentiment Analysis** (ML + Rule-based)
    - 🗣️ **Dialect Identification** (Traditional NLP)
    - 📝 **Text Summarization** (LLM-based)
    - 🧠 **RAG Question Answering** (Knowledge Base)
    """)
    
    st.markdown("---")
    st.markdown("## 📊 System Stats")
    st.metric("Tasks Supported", "5")
    st.metric("Models Used", "6+")
    st.metric("Languages", "Arabic, English")
    
    st.markdown("---")
    st.markdown("## 🔧 Technologies")
    st.markdown("""
    - **Traditional NLP**: CAMeL Tools, TF-IDF, Rule-based
    - **LLM**: OpenAI GPT-4o-mini, Transformers
    - **Embeddings**: Sentence Transformers
    - **UI**: Streamlit
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    task = st.selectbox(
        "📋 Select Task",
        [
            "🌍 Machine Translation",
            "😊 Sentiment Analysis",
            "🗣️ Dialect Identification",
            "📝 Text Summarization",
            "🧠 Question Answering (RAG)"
        ]
    )

with col2:
    if task == "🌍 Machine Translation":
        st.info("💡 Auto-detects language (Arabic/English)")

# User input
user_input = st.text_area(
    "📝 Enter your text:",
    height=150,
    placeholder="Type your text here...\n\nExample:\n- 'مرحبا كيف حالك؟'\n- 'This is amazing!'\n- 'What is the capital of Lebanon?'"
)

# Run button
if st.button("🚀 Run", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Processing..."):
            st.markdown("---")
            
            # ==================== MACHINE TRANSLATION ====================
            if task == "🌍 Machine Translation":
                lang = detect_language(user_input)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📤 Original Text")
                    st.info(user_input)
                    
                    if lang == "ar":
                        st.caption("🔤 Detected: Arabic")
                    else:
                        st.caption("🔤 Detected: English")
                
                with col2:
                    st.markdown("### 📥 Translated Text")
                    
                    if lang == "ar":
                        result = translate_text(user_input, "ar", "en")
                        st.success(f"**English:**\n{result}")
                    else:
                        result = translate_text(user_input, "en", "ar")
                        st.success(f"**Arabic:**\n{result}")
                    
                    st.caption("✨ Powered by MarianMT (Helsinki-NLP)")
            
            # ==================== SENTIMENT ANALYSIS ====================
            elif task == "😊 Sentiment Analysis":
                result = analyze_sentiment(user_input)
                label = result["label"]
                method = result["method"]
                confidence = result.get("confidence", 0.5)
                
                st.markdown("### 📊 Sentiment Analysis Result")
                
                # Display sentiment with colors and emojis
                if label == "positive":
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>😄 Positive Sentiment</h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Method:</strong> {method}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                    
                elif label == "negative":
                    st.markdown(f"""
                    <div class="error-box">
                        <h3>😡 Negative Sentiment</h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Method:</strong> {method}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.markdown(f"""
                    <div class="info-box">
                        <h3>😐 Neutral Sentiment</h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Method:</strong> {method}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # ==================== DIALECT IDENTIFICATION ====================
            elif task == "🗣️ Dialect Identification":
                result = identify_dialect(user_input)
                
                st.markdown("### 🗣️ Dialect Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Input Text")
                    st.info(user_input)
                
                with col2:
                    st.markdown("#### Detected Dialect")
                    if result["dialect"] == "MSA":
                        st.success(f"**{result['dialect']}** (Modern Standard Arabic)")
                    elif result["dialect"] == "Egyptian":
                        st.success(f"**{result['dialect']}** 🇪🇬")
                    elif result["dialect"] == "Levantine":
                        st.success(f"**{result['dialect']}** 🇱🇧🇸🇾🇯🇴🇵🇸")
                    elif result["dialect"] == "Gulf":
                        st.success(f"**{result['dialect']}** 🇸🇦🇦🇪🇰🇼🇶🇦")
                    elif result["dialect"] == "Maghrebi":
                        st.success(f"**{result['dialect']}** 🇲🇦🇩🇿🇹🇳🇱🇾")
                    else:
                        st.info(f"**{result['dialect']}**")
                    
                    st.caption(f"🔍 Method: {result['method']}")
                    st.caption(f"🎯 Confidence: {result['confidence']:.2%}")
            
            # ==================== TEXT SUMMARIZATION ====================
            elif task == "📝 Text Summarization":
                if len(user_input.split()) < 20:
                    st.warning("⚠️ Text is too short for summarization. Please enter at least 20 words.")
                else:
                    result = summarize_text(user_input)
                    
                    st.markdown("### 📝 Text Summarization")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Original Text")
                        st.info(user_input)
                        st.caption(f"📊 Length: {len(user_input.split())} words")
                    
                    with col2:
                        st.markdown("#### Summary")
                        st.success(result["summary"])
                        st.caption(f"📊 Length: {len(result['summary'].split())} words")
                        st.caption(f"✨ Method: {result['method']}")
                        
                        # Show compression ratio
                        original_len = len(user_input.split())
                        summary_len = len(result['summary'].split())
                        compression = (1 - summary_len/original_len) * 100
                        st.caption(f"📉 Compression: {compression:.1f}% reduction")
            
            # ==================== RAG QUESTION ANSWERING ====================
            elif task == "🧠 Question Answering (RAG)":
                try:
                    result = answer_question(user_input)
                    
                    st.markdown("### 🧠 Question Answering")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("#### Question")
                        st.info(user_input)
                    
                    with col2:
                        st.markdown("#### Answer")
                        st.success(result["answer"])
                    
                    # Show retrieved sources if available
                    if result.get("retrieved_chunks"):
                        with st.expander("📚 Sources & Context"):
                            for i, chunk in enumerate(result["retrieved_chunks"][:3], 1):
                                st.markdown(f"**Source {i}:** `{chunk['source']}`")
                                st.markdown(f"**Relevance Score:** {chunk['score']:.3f}")
                                st.markdown(f"**Context:** {chunk['text'][:300]}...")
                                st.divider()
                    
                except ValueError as e:
                    if "OPENAI_API_KEY" in str(e):
                        st.error("🔑 OpenAI API key not found. Please set your API key in environment variables.")
                        st.info("Run: `setx OPENAI_API_KEY 'your-key-here'` and restart terminal")
                    else:
                        st.error(f"Error: {str(e)}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("💡 Make sure you have documents in `data/kb/` folder")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>🤖 Arabic Virtual AI Assistant | Built with ❤️ using Traditional NLP + LLMs</p>",
    unsafe_allow_html=True
)