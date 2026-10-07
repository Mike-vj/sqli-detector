import pandas as pd

df = pd.read_csv("data/sqli.csv", encoding="utf-8", on_bad_lines="skip")

# Merge the misplaced label columns into one
df["Label"] = df["Label"].combine_first(df["Unnamed: 2"]).combine_first(df["Unnamed: 3"])

# Drop the now-redundant columns
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3"])

# Drop rows where Sentence or Label is still missing
df = df.dropna(subset=["Sentence", "Label"])

# Keep only rows where Label is actually 0 or 1 (drops rows still misaligned)
df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
df = df.dropna(subset=["Label"])
df = df[df["Label"].isin([0, 1])]

# Drop duplicate rows
df = df.drop_duplicates()

# Make sure Label is an integer 0/1
df["Label"] = df["Label"].astype(int)

print("Shape after cleaning:", df.shape)
print("\nClass balance:\n", df["Label"].value_counts())
print("\nSample:\n", df.sample(5))

df.to_csv("data/sqli_clean.csv", index=False)
print("\nSaved cleaned file to data/sqli_clean.csv")