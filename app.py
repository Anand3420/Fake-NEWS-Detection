import streamlit as st
import pickle
import re
import string
import pandas as pd
from datetime import datetime
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Fake News Detection", layout="wide")

# ===============================
# LOAD MODEL, TF-IDF & ACCURACY
# ===============================
model = pickle.load(open("model\model.pkl", "rb"))
tfidf = pickle.load(open("model\\tfidf.pkl", "rb"))

try:
    accuracy = pickle.load(open("model\accuracy.pkl", "rb"))
except:
    accuracy = None

# ===============================
# SESSION STATE
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# COUNTRY → COORDINATES
# ===============================
COUNTRY_COORDS = {
    "India": [20.5937, 78.9629],
    "United Kingdom": [55.3781, -3.4360],
    "United States": [37.0902, -95.7129],
    "Russia": [61.5240, 105.3188],
    "Ukraine": [48.3794, 31.1656],
    "China": [35.8617, 104.1954],
    "Australia": [-25.2744, 133.7751],
    "Canada": [56.1304, -106.3468],
    "Germany": [51.1657, 10.4515],
    "France": [46.2276, 2.2137],
    "Japan": [36.2048, 138.2529]
}

# ===============================
# TEXT CLEANING
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# ===============================
# TITLE
# ===============================
st.title("📰 Fake News Detection Website")
st.write("Predict whether a news article is **REAL**, **FAKE**, or **UNCERTAIN**.")

# ===============================
# TABS
# ===============================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📰 Prediction", "📊 Analytics", "🕒 History", "🗺️ Map", "ℹ️ About"]
)

# =====================================================
# TAB 1 — PREDICTION
# =====================================================
with tab1:
    st.subheader("📰 News Prediction")

    news_input = st.text_area("Enter News Text Here:")

    if st.button("Predict"):
        if news_input.strip() == "":
            st.warning("⚠️ Please enter some news text")
        else:
            with st.spinner("🔍 Analyzing news..."):
                cleaned = clean_text(news_input)
                vector = tfidf.transform([cleaned])

                probs = model.predict_proba(vector)[0]
                real_prob = probs[0] * 100
                fake_prob = probs[1] * 100

                if real_prob >= 60:
                    result = "REAL"
                    st.success(f"✅ REAL NEWS (Confidence: {real_prob:.2f}%)")
                elif fake_prob >= 60:
                    result = "FAKE"
                    st.error(f"❌ FAKE NEWS (Confidence: {fake_prob:.2f}%)")
                else:
                    result = "UNCERTAIN"
                    st.warning(
                        f"⚠️ UNCERTAIN NEWS\n"
                        f"Real: {real_prob:.2f}% | Fake: {fake_prob:.2f}%"
                    )

                # Save history
                st.session_state.history.append({
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Prediction": result,
                    "Real %": round(real_prob, 2),
                    "Fake %": round(fake_prob, 2),
                    "Text": news_input[:80] + "..."
                })

# =====================================================
# TAB 2 — ANALYTICS (SAFE IMAGE LOADING)
# =====================================================
with tab2:
    st.subheader("📊 Model Analytics")

    if accuracy is not None:
        st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png", caption="Confusion Matrix")
    else:
        st.info("Confusion Matrix not found. Please run train_model.py")

    if os.path.exists("roc_curve.png"):
        st.image("roc_curve.png", caption="ROC Curve")
    else:
        st.info("ROC Curve not found. Please run train_model.py")

# =====================================================
# TAB 3 — HISTORY
# =====================================================
with tab3:
    st.subheader("🕒 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.history))

        if st.button("Clear History"):
            st.session_state.history = []
            st.success("History cleared.")

# =====================================================
# TAB 4 — MAP
# =====================================================
with tab4:
    st.subheader("🗺️ News Location Map")

    country = st.selectbox(
        "Select Country related to the news",
        ["Select"] + sorted(COUNTRY_COORDS.keys())
    )

    if country != "Select":
        lat, lon = COUNTRY_COORDS[country]
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))
        st.success(f"📍 Showing location for: {country}")
    else:
        st.info("Please select a country to display on the map.")

# =====================================================
# TAB 5 — ABOUT
# =====================================================
with tab5:
    st.subheader("ℹ️ About This Project")

    st.markdown(
        """
        **Fake News Detection System**

        - Final Year Engineering Project  
        - NLP + Machine Learning based  
        - TF-IDF + Naive Bayes Classifier  

        **Features**
        - Prediction with confidence  
        - Analytics (Accuracy, Confusion Matrix, ROC)  
        - Prediction History  
        - Country-wise Map Visualization  

        **Developer:** Anand  
        **Year:** 2025
        """
    )
