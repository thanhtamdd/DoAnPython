import joblib

model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_intent(text):
    text_tfidf = vectorizer.transform([text])
    intent = model.predict(text_tfidf)[0]
    return intent
    


if __name__ == "__main__":
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        intent = predict_intent(user_input)
        print("→ Intent dự đoán:", intent)
