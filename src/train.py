import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import normalize_query

df = pd.read_csv("data/sqli_clean.csv")

X = df["Sentence"].astype(str)
X = X.apply(normalize_query)
y = df["Label"]

# Stratified split keeps the same class ratio in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Character n-grams: SQLi depends on symbols (', --, =), not words
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("=== Logistic Regression ===")
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train_vec, y_train)
lr_preds = lr.predict(X_test_vec)
print(classification_report(y_test, lr_preds))
print(confusion_matrix(y_test, lr_preds))

print("\n=== Naive Bayes ===")
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
nb_preds = nb.predict(X_test_vec)
print(classification_report(y_test, nb_preds))
print(confusion_matrix(y_test, nb_preds))

# Save the best-looking model + vectorizer for the dashboard later
joblib.dump(lr, "models/lr_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("\nSaved Logistic Regression model and vectorizer to models/")