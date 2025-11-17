# train_classifier.py
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# -----------------------
# Helper text preprocessing
# -----------------------
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    # remove punctuation except spaces
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# -----------------------
# Load dataset
# -----------------------
IN_PATH = "item_database.xlsx"   # place your file here
df = pd.read_excel(IN_PATH)

# Expecting columns: "Item Name", "Category"
required_cols = ["Item Name", "Category"]
if not all(c in df.columns for c in required_cols):
    raise SystemExit(f"Required columns missing. Found: {df.columns.tolist()}")

# Drop rows with missing category or item name
df = df.dropna(subset=["Item Name", "Category"]).reset_index(drop=True)

# Clean text
df['item_text'] = df['Item Name'].astype(str).apply(clean_text)
df['category'] = df['Category'].astype(str).str.strip()

# Optional: look at class balance
class_counts = df['category'].value_counts()
print("Class distribution:\n", class_counts.head(30))

# If classes with very few samples exist, you can consider grouping or labeling them as 'Other'
# For now we'll keep as-is.

# -----------------------
# Train/test split
# -----------------------
X = df['item_text'].values
y = df['category'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -----------------------
# Vectorize text (TF-IDF)
# -----------------------
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)  # tune as needed
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -----------------------
# Label encoding
# -----------------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# -----------------------
# Train classifier (Multinomial Naive Bayes)
# -----------------------
clf = MultinomialNB(alpha=1.0)
clf.fit(X_train_tfidf, y_train_enc)

# -----------------------
# Evaluate model
# -----------------------
y_pred_enc = clf.predict(X_test_tfidf)
y_pred = le.inverse_transform(y_pred_enc)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Optional: confusion matrix (verbose)
# cm = confusion_matrix(y_test_enc, y_pred_enc)
# print("Confusion matrix:\n", cm)

# -----------------------
# Save model, vectorizer, encoder
# -----------------------
joblib.dump(clf, "nb_item_classifier.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Saved: nb_item_classifier.pkl, tfidf_vectorizer.pkl, label_encoder.pkl")
