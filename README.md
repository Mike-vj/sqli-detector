SQL Injection Detection System

A machine learning-based tool that analyzes text input (such as a database query or user-submitted field) and predicts whether it looks like a SQL Injection (SQLi) attempt, with a Streamlit web dashboard for live testing.

Project Goal

Traditional SQLi defenses often rely on hand-written regex/signature rules, which are brittle and easy to bypass once an attacker learns the pattern. This project explores whether a machine learning model, trained on character-level patterns in text, can generalize better than static rules while remaining fast enough for real-time use.

Dataset
Source: Kaggle "SQL Injection Dataset"
~30,000+ labeled text samples (Sentence = text, Label = 0 for benign, 1 for SQLi)
The raw file had misaligned columns on rows containing unescaped commas; these were repaired during cleaning. See src/clean.py.
Approach
Data cleaning (src/clean.py): merges misaligned label columns, drops unrecoverable/duplicate rows, saves data/sqli_clean.csv.
Preprocessing (src/preprocess.py): strips inline SQL comments (e.g. /* */) and normalizes whitespace, so obfuscated keyword-splitting tricks (e.g. UN/**/ION) are partially normalized before feature extraction.
Feature extraction: TF-IDF over character n-grams (2–5 chars) rather than word tokens, since SQLi relies on symbols and short patterns (', --, =, OR 1=1) that word-level tokenizing would destroy.
Model training (src/train.py): Logistic Regression and Multinomial Naive Bayes were trained and compared on a stratified 80/20 train/test split.
Dashboard (app.py): a Streamlit app that loads the saved model and lets a user type in text to get a live SQLi/benign verdict with a confidence score.
Results
Model	Accuracy	SQLi Recall	False Positives
Logistic Regression	99%	99%	5 / 3902
Naive Bayes	92%	98%	472 / 3902

Logistic Regression was selected for the dashboard — comparable recall to Naive Bayes but far fewer false positives, which matters for a usable security tool (too many false alarms cause alert fatigue and get ignored).

Known Limitation

A manual stress test (src/stress_test.py) beyond the dataset's test split revealed that payloads using inline SQL comments to split keywords (e.g. UN/**/ION SELECT username, password FROM users) can evade detection. The normalize_query() preprocessing step measurably improved the model's confidence on this payload (SQLi probability rose from 0.12 to 0.27) but did not fully close the gap, since the model was never trained on this exact cleaned phrasing.

This is a known limitation of character-n-gram-based detectors. A natural next step would be adding hand-crafted keyword features (e.g. counts of SELECT, UNION, OR, quote characters) alongside the TF-IDF representation, so the model isn't relying on character patterns alone.

Project Structure
sqli-detector/
├── data/
│   ├── sqli.csv              # raw dataset
│   └── sqli_clean.csv        # cleaned dataset
├── models/
│   ├── lr_model.pkl          # trained Logistic Regression model
│   └── vectorizer.pkl        # fitted TF-IDF vectorizer
├── src/
│   ├── clean.py              # data cleaning
│   ├── preprocess.py         # shared text normalization
│   ├── train.py              # training + evaluation
│   └── stress_test.py        # manual adversarial test cases
├── app.py                    # Streamlit dashboard
└── requirements.txt
How to Run
Create and activate a virtual environment, then install dependencies:
bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
(Optional — models are already trained and saved) Retrain from scratch:
bash
   python src/clean.py
   python src/train.py
Launch the dashboard:
bash
   streamlit run app.py

Then open the local URL Streamlit prints (usually http://localhost:8501).

Tech Stack

Python, pandas, scikit-learn (TF-IDF, Logistic Regression, Naive Bayes), joblib, Streamlit.