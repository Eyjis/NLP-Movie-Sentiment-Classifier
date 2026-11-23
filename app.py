import os
import pickle


def temporary_predict(review_text):
    """
    Temporary prediction function used until trained models are added.
    """
    text = review_text.lower()
    positive_words = [
        "good", "great", "amazing", "excellent", "love",
        "wonderful", "best", "fun", "enjoyed", "brilliant"
    ]
    negative_words = [
        "bad", "boring", "terrible", "awful", "worst",
        "waste", "poor", "hate", "dull", "disappointing"
    ]

    pos_score = sum(word in text for word in positive_words)
    neg_score = sum(word in text for word in negative_words)

    if pos_score >= neg_score:
        return "Positive"
    return "Negative"


def predict_sentiment(review_text, model_choice):
    """
    Loads the trained model from 'models/' if available;
    otherwise uses temporary_predict() seamlessly.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if model_choice == "1":
        model_path = os.path.join(base_dir, "models", "unigram_model.pkl")
        model_name = "Unigram Naive Bayes"
    else:
        model_path = os.path.join(base_dir, "models", "bigram_model.pkl")
        model_name = "Bigram (Unigram + Bigram) Naive Bayes"

    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            loaded_obj = pickle.load(file)

        if hasattr(loaded_obj, "predict"):
            raw_pred = loaded_obj.predict([review_text])[0]
        elif isinstance(loaded_obj, tuple) and len(loaded_obj) == 2:
            vectorizer, classifier = loaded_obj
            features = vectorizer.transform([review_text])
            raw_pred = classifier.predict(features)[0]
        elif isinstance(loaded_obj, dict) and "model" in loaded_obj and "vectorizer" in loaded_obj:
            features = loaded_obj["vectorizer"].transform([review_text])
            raw_pred = loaded_obj["model"].predict(features)[0]
        else:
            raise ValueError("Unsupported model format inside .pkl file.")

        label = "Positive" if str(raw_pred).lower() in ["pos", "positive", "1"] else "Negative"
        return label, f"{model_name} [Trained Model]"

    label = temporary_predict(review_text)
    return label, f"{model_name} [Temporary Mode]"


def main():
    print("=" * 55)
    print("   Movie Review Sentiment Classifier (NLP Project)")
    print("=" * 55)

    while True:
        print("\nChoose a model:")
        print("  [1] Unigram Naive Bayes")
        print("  [2] Bigram Naive Bayes")
        print("  [q] Exit")
        choice = input("Enter choice (1, 2, or q): ").strip().lower()

        if choice == "q":
            print("Program closed. Goodbye!")
            break

        if choice not in ["1", "2"]:
            print("Invalid choice. Please enter 1, 2, or q.")
            continue

        review = input("\nEnter an English movie review:\n> ").strip()
        if not review:
            print("Review text cannot be empty. Please try again.")
            continue

        prediction, mode_info = predict_sentiment(review, choice)
        print("-" * 55)
        print(f"Model      : {mode_info}")
        print(f"Prediction : {prediction}")
        print("-" * 55)


if __name__ == "__main__":
    main()
