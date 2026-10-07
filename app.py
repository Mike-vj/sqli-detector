import streamlit as st
import joblib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocess import normalize_query

st.set_page_config(page_title="SQL Injection Detector", page_icon="🛡️")

@st.cache_resource
def load_model():
    model = joblib.load("models/lr_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

st.title("🛡️ SQL Injection Detection System")
st.write("Enter a query or input string below to check if it looks like a SQL injection attempt.")

user_input = st.text_area("Input to analyze:", height=100, placeholder="e.g. SELECT * FROM users WHERE id = 1")

if st.button("Analyze") and user_input.strip():
    clean_text = normalize_query(user_input)
    vec = vectorizer.transform([clean_text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]

    if pred == 1:
        st.error(f"⚠️ Likely SQL Injection detected (confidence: {prob:.0%})")
    else:
        st.success(f"✅ Looks benign (SQLi probability: {prob:.0%})")

    st.progress(float(prob))

    with st.expander("Details"):
        st.write("**Original input:**", user_input)
        st.write("**Normalized input:**", clean_text)
        st.write("**Raw probability (SQLi):**", f"{prob:.4f}")

st.divider()
st.caption("Built with a Logistic Regression model trained on TF-IDF character n-grams. "
           "Known limitation: payloads using inline comments to split keywords (e.g. UN/**/ION) "
           "may sometimes evade detection.")