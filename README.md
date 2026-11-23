# Movie Review Sentiment Classifier: Comparing Bag-of-Words and Bigram Features with Naive Bayes

**Course:** CS1763 – Natural Language Processing  
**Institution:** Faculty of Computing and Information, Al-Baha University  
**Instructor:** Dr. Mohammed Alliheedi  

---

## 1. Project Overview
This project builds a supervised machine learning classifier to automatically label English movie reviews as **Positive** or **Negative**. Specifically, we conduct a controlled **one-change feature comparison** using **Multinomial Naive Bayes** to evaluate how adding **Bigram** features (`Unigram + Bigram`) impacts classification performance compared to standard **Bag-of-Words (`Unigram`)** features, alongside a **Majority-Class Baseline** and a lexicon-based **VADER** comparison.

---

## 2. Team Members and Task Distribution
1. **Ibrahem Khaled AL-Ghamdi (`445022298`)** – *Data Preparation & Dataset Analysis*: Loading NLTK movie reviews, label/duplicate checks, text cleaning, and generating the fixed stratified 80/20 train/test split (`random_state=42`).
2. **Abdullah Ibrahim Alzahrani (`445007956`)** – *Baseline & Unigram Naive Bayes*: Implementing the majority-class baseline, training the Unigram Multinomial Naive Bayes model, smoothing experiments, and extracting informative words.
3. **Yazeed Alzahrani (`445006893`)** – *Bigram Feature Engineering & Model*: Implementing Unigram + Bigram feature extraction, training the Bigram Naive Bayes model, and conducting feature extraction experiments.
4. **Mohammed Abdullah Al-Zahrani (`445022413`)** – *Evaluation Framework & Error Analysis*: Computing Accuracy, Precision, Recall, F1-score, Confusion Matrices, model comparison tables, and categorizing at least 20 misclassified reviews (e.g., negation, sarcasm).
5. **Essa Fayez Al-Zahrani (`445008299`)** – *Prediction Interface & Project Documentation*: Developing the interactive prediction interface (`app.py`), evaluating VADER, organizing the GitHub repository, writing `README.md`, and structuring the final report and presentation slides.

---

## 3. Dataset Information & Terms of Use
* **Dataset:** NLTK `movie_reviews` corpus (Cornell Movie Review Polarity Dataset v2.0, Pang & Lee, 2002).
* **Size & Balance:** 2,000 English movie reviews (1,000 Positive, 1,000 Negative).
* **Version & Download Date:** Dataset v2.0 (downloaded via `nltk.download('movie_reviews')` in November 2026).
* **Terms of Use:** Used strictly for academic and educational research purposes with proper citation to Pang & Lee (2002).
* **Train/Test Split & Reproducibility:** Fixed **80/20 stratified split** (1,600 training reviews / 400 testing reviews) generated using `random_state = 42`. Hyperparameter tuning is restricted to cross-validation on the training set only; the test set is evaluated once at the end.

---

## 4. Repository Structure
```text
NLP_Movie_Sentiment/
├── data/                  # Cleaned dataset and fixed 80/20 train/test split files
├── models/                # Saved trained models (unigram_model.pkl, bigram_model.pkl)
├── notebooks/             # Jupyter notebooks for cleaning, modeling, evaluation, and error analysis
├── app.py                 # Interactive CLI interface for predicting sentiment on new reviews
└── README.md              # Project documentation and execution instructions
