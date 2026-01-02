import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

st.set_page_config(page_title="SMS Spam Detector", page_icon="📱", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('spam_model.h5')
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_resources()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error: {e}")
    model_loaded = False

# Header
st.markdown('<div class="main-header"><h1>📱 SMS Spam Detection System</h1><p>Deep Learning Model | 97%+ Accuracy</p></div>', 
            unsafe_allow_html=True)

# Main UI
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Test SMS Message")
    
    user_input = st.text_area(
        "Enter message to analyze:",
        placeholder="Example: Congratulations! You won $1000...",
        height=120
    )
    
    if st.button("🚀 Analyze Message", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("⚠️ Please enter a message!")
        elif not model_loaded:
            st.error("❌ Model not loaded")
        else:
            with st.spinner("🔄 Analyzing..."):
                # Vectorize input
                vec = vectorizer.transform([user_input]).toarray()
                prediction = model.predict(vec, verbose=0)[0][0]
            
            st.divider()
            
            if prediction > 0.5:
                st.error(f"### 🚨 SPAM DETECTED")
                st.metric("Spam Confidence", f"{prediction*100:.1f}%")
                st.warning("""
                **⚠️ Warning:**
                - May be fraudulent
                - Don't click links
                - Don't share personal info
                """)
            else:
                st.success(f"### ✅ LEGITIMATE MESSAGE")
                st.metric("Ham Confidence", f"{(1-prediction)*100:.1f}%")
                st.info("✓ Safe message")
            
            # Fixed: Convert to Python float
            st.progress(float(prediction), text=f"Spam Score: {prediction*100:.1f}%")


with col2:
    st.subheader("📈 Model Stats")
    st.metric("Accuracy", "97%+")
    st.metric("Total Messages", "5,572")
    st.metric("Training Time", "~5 min")

st.divider()
st.subheader("📝 Examples")

col_ex1, col_ex2 = st.columns(2)

with col_ex1:
    st.markdown("#### ✅ **HAM**")
    for ex in ["Hey, are you free?", "Meeting at 3 PM", "Mom: Coming home?"]:
        st.text(f"• {ex}")

with col_ex2:
    st.markdown("#### 🚨 **SPAM**")
    for ex in ["You won £1000!", "URGENT: Verify now", "FREE money!"]:
        st.text(f"• {ex}")

st.divider()
st.markdown("<div style='text-align: center; color: #666;'><p>🎓 College Project | SMS Spam Detection</p></div>", unsafe_allow_html=True)
