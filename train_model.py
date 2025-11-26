# train_model.py
import pandas as pd
import random
import re
import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

def clean_text(text):
    text = str(text).lower()
    text = unidecode.unidecode(text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

# Paths
TRAIN_CSV = Path("train_data.csv")
RESP_CSV = Path("responses.csv")
OUT_RESP_AUG = Path("responses_augmented.csv")
OUT_TRAIN_AUG = Path("train_data_augmented.csv")

# Load datasets
if not TRAIN_CSV.exists():
    raise FileNotFoundError("Please create train_data.csv with columns: intent,text")
if not RESP_CSV.exists():
    raise FileNotFoundError("Please create responses.csv with columns: intent,response")

df = pd.read_csv(str(TRAIN_CSV))
responses_df = pd.read_csv(str(RESP_CSV))

# augmentation settings
prefixes = ["Mình muốn biết ", "Cho mình hỏi ", "Xin hỏi ", "Bạn cho mình hỏi "]
suffixes = [" được không?", " giúp mình nhé.", " nhé!", "?"]
synonyms = {
    "giá": ["giá tiền", "chi phí", "mức giá", "bao nhiêu"],
    "sản phẩm": ["mặt hàng", "đồ", "món hàng", "hàng hóa"],
    "mua": ["tậu", "sắm", "đặt hàng", "order"],
    "quà": ["quà tặng", "món quà", "tặng phẩm"],
    "xem": ["cho xem", "xem thử", "tham khảo"]
}

def generate_variants(text, n=5):
    variants = set([text])
    for _ in range(n):
        new_text = text
        for word, syns in synonyms.items():
            if word in new_text:
                new_text = new_text.replace(word, random.choice(syns))
        new_text = random.choice(prefixes) + new_text + random.choice(suffixes)
        variants.add(new_text)
    return list(variants)

# Build augmented train data
aug = []
for _, row in df.iterrows():
    text = row["text"]
    intent = row["intent"]
    for v in generate_variants(text, n=8):
        aug.append({"text": v, "intent": intent})

df_aug = pd.DataFrame(aug)
df_aug["clean_text"] = df_aug["text"].apply(clean_text)
df_aug.to_csv(str(OUT_TRAIN_AUG), index=False)
print("Saved augmented train:", OUT_TRAIN_AUG)

# Vectorize & train
X = df_aug["clean_text"]
y = df_aug["intent"]
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=2000, solver="liblinear")
model.fit(X_train, y_train)

print("=== Classification report ===")
print(classification_report(y_test, model.predict(X_test)))

# Save model & vectorizer
joblib.dump(model, "intent_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Saved intent_model.pkl and vectorizer.pkl")

# Augment responses (simple variants)
def generate_response_variants(text, n=3):
    variants = set([text])
    for _ in range(n):
        new_text = text
        for word, syns in synonyms.items():
            if word in new_text:
                new_text = new_text.replace(word, random.choice(syns))
        new_text = new_text + random.choice([" 😊","!", " nhé!"])
        variants.add(new_text)
    return list(variants)

aug_res = []
for _, row in responses_df.iterrows():
    intent = row["intent"]
    resp = row["response"]
    for v in generate_response_variants(resp, n=4):
        aug_res.append({"intent": intent, "response": v})

pd.DataFrame(aug_res).to_csv(str(OUT_RESP_AUG), index=False)
print("Saved responses augmented:", OUT_RESP_AUG)
