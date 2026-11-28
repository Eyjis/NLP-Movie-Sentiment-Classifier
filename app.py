import os
import warnings
import joblib
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "unigram": os.path.join(BASE_DIR, "models", "unigram_model.pkl"),
    "bigram": os.path.join(BASE_DIR, "models", "bigram_model.pkl"),
}

LABEL_MAP = {
    "pos": "Positive",
    "positive": "Positive",
    "neg": "Negative",
    "negative": "Negative",
}


def load_models(paths):
    """تحميل النماذج المدربة من مجلد models/ والتحقق من وجودها."""
    models = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ملف النموذج غير موجود: {path}\n"
                "تأكد من وضع unigram_model.pkl و bigram_model.pkl داخل مجلد models/"
            )
        models[name] = joblib.load(path)
    return models


def normalize_label(raw_label):
    """تحويل مخرجات التصنيف إلى صيغة موحدة (Positive / Negative)."""
    return LABEL_MAP.get(str(raw_label).strip().lower(), str(raw_label))


def predict_review(model, text):
    """تنفيذ التنبؤ وحساب احتمالات الفئات واستخراج أبرز الميزات المؤثرة في النص."""
    vectorizer = model.steps[0][1]
    classifier = model.steps[1][1]

    X = vectorizer.transform([text])
    raw_pred = classifier.predict(X)[0]
    probs = classifier.predict_proba(X)[0]

    label = normalize_label(raw_pred)
    confidence = float(np.max(probs)) * 100

    classes_norm = [normalize_label(c) for c in classifier.classes_]
    pos_idx = classes_norm.index("Positive")
    neg_idx = classes_norm.index("Negative")

    feature_names = vectorizer.get_feature_names_out()
    nonzero_indices = X.nonzero()[1]
    if len(nonzero_indices) > 0:
        log_odds = (
            classifier.feature_log_prob_[pos_idx, nonzero_indices]
            - classifier.feature_log_prob_[neg_idx, nonzero_indices]
        )
        ranked = sorted(
            zip(feature_names[nonzero_indices], log_odds),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:4]
        top_cues = [f"{word} ({score:+.2f})" for word, score in ranked]
    else:
        top_cues = ["No known vocabulary features matched"]

    return {
        "label": label,
        "confidence": confidence,
        "pos_prob": float(probs[pos_idx]) * 100,
        "neg_prob": float(probs[neg_idx]) * 100,
        "top_cues": top_cues,
    }


def display_result(title, result):
    print(f"\n--- {title} ---")
    print(f"Sentiment  : {result['label']} ({result['confidence']:.2f}% confidence)")
    print(f"Probabilities: Positive = {result['pos_prob']:.2f}% | Negative = {result['neg_prob']:.2f}%")
    print(f"Key Features : {', '.join(result['top_cues'])}")


def main():
    print("=" * 60)
    print("Movie Review Sentiment Classifier (Unigram vs. Bigram)")
    print("=" * 60)

    try:
        models = load_models(MODEL_PATHS)
        print("Models loaded successfully from 'models/'.\n")
    except FileNotFoundError as err:
        print(f"[Error] {err}")
        return

    while True:
        print("\nChoose mode:")
        print("  [1] Unigram Model (Member 2 - Abdullah)")
        print("  [2] Bigram Model  (Member 3 - Yazeed)")
        print("  [3] Compare Both Models Side-by-Side")
        print("  [q] Quit")

        choice = input("\nEnter your choice (1/2/3/q): ").strip().lower()
        if choice == "q":
            print("Exiting Sentiment Classifier. Goodbye!")
            break

        if choice not in {"1", "2", "3"}:
            print("Invalid choice. Please enter 1, 2, 3, or q.")
            continue

        review = input("\nEnter movie review text: ").strip()
        if not review:
            print("Review text cannot be empty. Please try again.")
            continue

        if choice == "1":
            res_u = predict_review(models["unigram"], review)
            display_result("Unigram Model (CountVectorizer + MultinomialNB)", res_u)

        elif choice == "2":
            res_b = predict_review(models["bigram"], review)
            display_result("Bigram Model (Unigram+Bigram + MultinomialNB)", res_b)

        elif choice == "3":
            res_u = predict_review(models["unigram"], review)
            res_b = predict_review(models["bigram"], review)
            display_result("Unigram Model (Member 2)", res_u)
            display_result("Bigram Model  (Member 3)", res_b)

            print("\n--- Comparison Summary ---")
            if res_u["label"] == res_b["label"]:
                print(f"Agreement: Both models predict [{res_b['label']}].")
            else:
                print(
                    f"Disagreement: Unigram predicts [{res_u['label']}] "
                    f"while Bigram predicts [{res_b['label']}]."
                )


if __name__ == "__main__":
    main()
