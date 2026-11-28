# Movie Review Sentiment Classifier (Unigram vs. Bigram)

An end-to-end Natural Language Processing (NLP) project that classifies movie reviews into **Positive** or **Negative** sentiment using Multinomial Naive Bayes, comparing a baseline **Unigram** representation against an engineered **Unigram + Bigram** feature pipeline.

---

## Project Structure

```text
NLP-Movie-Sentiment-Classifier/
├── data/
│   ├── bigram_experiment_results.csv   # 36-run 5-fold CV hyperparameter grid search results
│   └── bigram_predictions.csv          # Test set predictions (400 reviews)
├── models/
│   ├── unigram_model.pkl               # Trained Unigram pipeline (Member 2)
│   └── bigram_model.pkl                # Trained Unigram+Bigram pipeline (Member 3)
├── notebooks/
│   └── bigram_model.ipynb              # Bigram feature engineering & tuning notebook
├── app.py                              # Interactive CLI prediction & comparison tool
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
