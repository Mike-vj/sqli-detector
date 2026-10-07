import joblib
from preprocess import normalize_query

model = joblib.load("models/lr_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

test_cases = [
    ("1' OR '1'='1", "SQLi - classic"),
    ("admin'--", "SQLi - comment trick"),
    ("1; DROP TABLE users;--", "SQLi - stacked query"),
    ("UN/**/ION SELECT username, password FROM users", "SQLi - obfuscated UNION"),
    ("SeLeCt * fRoM users WHERE id=1 OR 1=1", "SQLi - mixed case"),
    ("john.smith@example.com", "benign - email"),
    ("SELECT name FROM products WHERE price < 100", "benign - normal query text"),
    ("I'd like to order a large pizza", "benign - everyday text"),
    ("O'Brien's Pub on 5th street", "benign - apostrophe, no SQLi"),
]

for text, label in test_cases:
    clean_text = normalize_query(text)
    vec = vectorizer.transform([clean_text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]  # probability of being SQLi
    verdict = "SQLi" if pred == 1 else "benign"
    print(f"[{verdict:6}] (p={prob:.2f})  expected: {label:35}  input: {text}")