import pandas as pd
import re
import string
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# ===============================
# 1. LOAD DATASETS
# ===============================
bbc = pd.read_csv("datasets\\bbc_news.csv", on_bad_lines="skip", engine="python")
real = pd.read_csv("datasets\True.csv")
fake = pd.read_csv("datasets\Fake.csv", on_bad_lines="skip", engine="python")


# BBC → REAL (0)
bbc["text"] = bbc["title"].fillna("") + " " + bbc["description"].fillna("")
bbc["label"] = 0

# REAL -> REAL(0)

real["text"] = real["title"].fillna("") + " " + real["text"].fillna("")
real["label"] = 0

#combined dataset bbc + real 
combined_dataset = pd.DataFrame()
combined_dataset["text"] = pd.concat([bbc["text"], real["text"]])
combined_dataset["label"] = pd.concat([bbc["label"], real["label"]])

# Fake → FAKE (1)
fake["label"] = 1

bbc = bbc[["text", "label"]].dropna()
fake = fake[["text", "label"]].dropna()

# ===============================
# 2. DATASET STATISTICS
# ===============================
stats = {
    "Total Samples": len(combined_dataset) + len(fake),
    "Real News": len(combined_dataset),
    "Fake News": len(fake)
}

print("\n📊 DATASET STATISTICS")
for k, v in stats.items():
    print(f"{k}: {v}")

pickle.dump(stats, open("data_stats.pkl", "wb"))

# ===============================
# 3. COMBINE & SHUFFLE
# ===============================
data = pd.concat([combined_dataset, fake], axis=0)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# ===============================
# 4. TEXT CLEANING
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data["clean_text"] = data["text"].apply(clean_text)

# ===============================
# 5. TRAIN / TEST SPLIT
# ===============================
X = data["clean_text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 6. TF-IDF + NAIVE BAYES MODEL
# ===============================
tfidf = TfidfVectorizer(stop_words="english", max_df=0.8)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = MultinomialNB(alpha=0.5)
model.fit(X_train_tfidf, y_train)

# ===============================
# 7. ACCURACY
# ===============================
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ MODEL ACCURACY: {accuracy*100:.2f}%")
pickle.dump(accuracy, open(r"model\accuracy.pkl", "wb"))

# ===============================
# 8. CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, y_pred)
pickle.dump(cm, open(r"model\confusion_matrix.pkl", "wb"))

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("📉 confusion_matrix.png generated")

# ===============================
# 9. ROC CURVE
# ===============================
y_prob = model.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

pickle.dump((fpr, tpr, roc_auc), open("roc_data.pkl", "wb"))

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

print("📈 roc_curve.png generated")

# ===============================
# 10. SAVE MODEL & VECTORIZER
# ===============================
pickle.dump(model, open(r"model\model.pkl", "wb"))
pickle.dump(tfidf, open(r"model\tfidf.pkl", "wb"))

print("\n✅ TRAINING COMPLETE — ALL FILES READY FOR app.py")
