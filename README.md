# Movie Review Sentiment Classifier


**Course:** CS1763 - Natural Language Processing  
**University:** Al-Baha University, Faculty of Computing and Information  
**Instructor:** Dr. Mohammed Alliheedi  


This project classifies English movie reviews as **Positive** or **Negative** using Multinomial Naive Bayes. The main goal is to compare standard Bag-of-Words (**Unigram**) features against **Bigram** (`Unigram + Bigram`) features under the exact same setup. We also include a Majority-Class baseline and compare our results with NLTK's rule-based **VADER** analyzer.


1. **Ibrahem Khaled AL-Ghamdi (`445022298`)** - Data loading, text cleaning, duplicate checks, and creating the fixed 80/20 train/test split (`random_state=42`).
2. **Abdullah Ibrahim Alzahrani (`445007956`)** - Majority-class baseline, Unigram Naive Bayes model, smoothing tests, and top unigram words.
3. **Yazeed Alzahrani (`445006893`)** - Bigram feature extraction (`ngram_range=(1, 2)`), Bigram Naive Bayes model, and top bigram phrases.
4. **Mohammed Abdullah Al-Zahrani (`445022413`)** - Model evaluation (Accuracy, Precision, Recall, F1, Confusion Matrices) and error analysis on 20+ misclassified reviews.
5. **Essa Fayez Al-Zahrani (`445008299`)** - CLI prediction tool (`app.py`), VADER comparison test, GitHub setup, documentation, and report/slides structure.


* **Source:** NLTK `movie_reviews` corpus (Cornell Movie Review Polarity Dataset v2.0, Pang & Lee, 2002).
* **Size:** 2,000 reviews total (1,000 Positive, 1,000 Negative).
* **Version & Date:** Dataset v2.0, downloaded via NLTK in November 2026.
* **Usage:** Used strictly for academic coursework with citation to Pang & Lee (2002).
* **Split:** Fixed 80/20 stratified split (1,600 train / 400 test) using `random_state = 42`. All tuning is done using cross-validation on the training set only.


```text
NLP_Movie_Sentiment/
├── data/          # Cleaned dataset and train/test split files
├── models/        # Saved models (unigram_model.pkl, bigram_model.pkl)
├── notebooks/     # Colab/Jupyter notebooks for cleaning, models, and evaluation
├── app.py         # Command-line tool to test new movie reviews
└── README.md      # Project info and run instructions
