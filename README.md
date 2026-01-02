# 📱 SMS Spam Detection System

A deep learning-based SMS spam detection application built with Streamlit and TensorFlow.

## 🚀 Features
- Real-time SMS analysis
- Deep Learning model with 97%+ accuracy
- User-friendly Streamlit interface
- Spam/Ham confidence metrics

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MP-3-2
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

## 📂 Project Structure
- `app.py`: Main Streamlit application
- `spam_model.h5`: Pre-trained TensorFlow model
- `vectorizer.pkl`: Pickled TF-IDF vectorizer
- `requirements.txt`: Project dependencies
