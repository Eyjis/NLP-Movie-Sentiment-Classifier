# Movie Review Sentiment Classifier: Comparing Bag-of-Words and Bigram Features with Naive Bayes

A sentiment analysis project comparing **Unigram (Bag-of-Words)** and **Bigram** feature representations using Multinomial Naive Bayes on the NLTK Movie Reviews dataset.

## Team Members
* **Ibrahem Khaled AL-Ghamdi** — Data Preparation & Preprocessing
* **Abdullah Ibrahim Alzahrani** — Unigram Model Implementation
* **Yazeed Alzahrani** — Bigram Model Implementation
* **Mohammed Abdullah Al-Zahrani** — Model Evaluation & Error Analysis
* **Essa Fayez Al-Zahrani** — Prediction Interface (`app.py`) & Project Documentation

---

## Dataset Overview
The models were trained and evaluated on the **NLTK Movie Reviews Corpus** (2,000 total reviews balanced equally between positive and negative classes), split into **1,600 training reviews** and **400 test reviews** (200 positive, 200 negative).

---

## Repository Structure
```text
├── app.py                                      # Interactive CLI sentiment prediction tool
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation and final results
├── data/
│   ├── cleaned_reviews.csv                     # Sample preprocessed reviews
│   ├── unigram_predictions.csv                 # Unigram test set predictions (400 reviews)
│   ├── bigram_predictions.csv                  # Bigram test set predictions (400 reviews)
│   ├── comparison_results.csv                  # Summary evaluation metrics
│   ├── error_analysis_results.csv              # All 78 misclassified reviews with error cues
│   ├── unigram_only_errors.csv                 # 22 reviews misclassified only by Unigram
│   ├── bigram_only_errors.csv                  # 6 reviews misclassified only by Bigram
│   ├── error_analysis_sample_20_annotated.csv  # Manually annotated sample of 20 errors
│   └── manual_error_summary.csv                # Breakdown of manual error categories
├── models/                                     # Saved Unigram and Bigram Naive Bayes models
├── notebooks/
│   ├── evaluation.ipynb                        # Model comparison and confusion matrices
│   └── error_analysis.ipynb                    # Automated and manual error analysis
└── results/
    ├── confusion_matrices_comparison.png       # Side-by-side confusion matrices
    └── manual_error_types_distribution.png     # Distribution of manual error categories
