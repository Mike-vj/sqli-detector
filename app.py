import os
import sys
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocess import normalize_query

st.set_page_config(page_title="SQLi Shield", page_icon="🛡️", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .hero {
        padding: 1.6rem 2rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #2563eb 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p  { margin: .3rem 0 0 0; opacity: .85; }
    .verdict {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        font-size: 1.15rem;
        font-weight: 600;
        margin: .5rem 0 1rem 0;
    }
    .verdict.bad  { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .verdict.warn { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
    .verdict.good { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ SQLi Shield</h1>
        <p>Machine-learning SQL injection detection &middot;
        TF-IDF character n-grams + Logistic Regression</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Model ----------
@st.cache_resource
def load_model():
    return joblib.load("models/lr_model.pkl"), joblib.load("models/vectorizer.pkl")


model, vectorizer = load_model()


def score(text: str) -> float:
    clean = normalize_query(text)
    return float(model.predict_proba(vectorizer.transform([clean]))[0][1])


def risk_level(p: float, threshold: float):
    if p >= max(threshold, 0.75):
        return "HIGH RISK", "bad", "🚨"
    if p >= threshold:
        return "SUSPICIOUS", "warn", "⚠️"
    return "LOOKS SAFE", "good", "✅"


# ---------- Sidebar ----------
EXAMPLES = {
    "SQLi: classic": "1' OR '1'='1",
    "SQLi: comment trick": "admin'--",
    "SQLi: stacked query": "1; DROP TABLE users;--",
    "SQLi: obfuscated UNION": "UN/**/ION SELECT username, password FROM users",
    "Benign: email": "john.smith@example.com",
    "Benign: apostrophe": "O'Brien's Pub on 5th street",
}

st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider(
    "Detection threshold", 0.10, 0.90, 0.50, 0.05,
    help="Lower = catches more attacks but raises false alarms.",
)
st.sidebar.header("🧪 Try an example")
choice = st.sidebar.selectbox("Example input", ["(type my own)"] + list(EXAMPLES))

if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Tabs ----------
tab_scan, tab_batch, tab_about = st.tabs(["🔍 Analyze", "📂 Batch scan", "ℹ️ About"])

# ----- Single analysis -----
with tab_scan:
    default = "" if choice == "(type my own)" else EXAMPLES[choice]
    user_input = st.text_area(
        "Input to analyze", value=default, height=110,
        placeholder="e.g. SELECT * FROM users WHERE id = 1",
    )

    if st.button("Analyze", type="primary") and user_input.strip():
        p = score(user_input)
        label, css, icon = risk_level(p, threshold)

        st.markdown(
            f'<div class="verdict {css}">{icon} {label} &mdash; SQLi probability {p:.0%}</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("SQLi probability", f"{p:.1%}")
        c2.metric("Threshold", f"{threshold:.0%}")
        c3.metric("Input length", f"{len(user_input)} chars")
        st.progress(min(max(p, 0.0), 1.0))

        with st.expander("Details"):
            st.write("**Original input:**", user_input)
            st.write("**Normalized input:**", normalize_query(user_input))
            st.write("**Raw probability:**", f"{p:.4f}")

        st.session_state.history.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "input": user_input[:80],
            "probability": round(p, 3),
            "verdict": label,
        })

    if st.session_state.history:
        st.subheader("Recent checks")
        st.dataframe(pd.DataFrame(st.session_state.history[:10]),
                     use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ----- Batch scan -----
with tab_batch:
    st.write("Upload a CSV and the app scores every row in a chosen text column.")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        df = pd.read_csv(file, on_bad_lines="skip")
        col = st.selectbox("Column containing the text", df.columns)
        if st.button("Scan file"):
            texts = df[col].astype(str).apply(normalize_query)
            df["sqli_probability"] = model.predict_proba(vectorizer.transform(texts))[:, 1]
            df["flagged"] = df["sqli_probability"] >= threshold

            m1, m2, m3 = st.columns(3)
            m1.metric("Rows scanned", len(df))
            m2.metric("Flagged as SQLi", int(df["flagged"].sum()))
            m3.metric("Flag rate", f"{df['flagged'].mean():.1%}")

            st.dataframe(df.sort_values("sqli_probability", ascending=False),
                         use_container_width=True, hide_index=True)
            st.download_button("Download results (CSV)",
                               df.to_csv(index=False).encode("utf-8"),
                               "sqli_scan_results.csv", "text/csv")

# ----- About -----
with tab_about:
    st.markdown(
        """
        **How it works**
        1. Input is normalized (inline `/* */` comments stripped, whitespace tidied).
        2. TF-IDF over character n-grams (2-5) turns the text into numbers.
        3. A Logistic Regression model outputs the probability of SQL injection.

        **Test-set results:** 99% accuracy, 5 false positives and 28 missed attacks
        out of 6,170 samples.

        **Known limitation:** payloads that split keywords with inline comments
        (e.g. `UN/**/ION`) can evade detection. Lowering the threshold catches more
        of them at the cost of extra false alarms.
        """
    )