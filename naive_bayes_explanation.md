Naive Bayes Sentiment Classifier

Overview

This document describes the classic machine learning baseline for our IMDB
sentiment classification project. We trained a Multinomial Naive Bayes
classifier on TF-IDF features to predict whether a movie review expresses a
positive or negative sentiment. Naive Bayes was chosen as the classic
counterpart to the deep learning model (LSTM) because it is fast, simple, and a
well-established strong baseline for text classification.

The implementation reads the already-cleaned dataset (IMDB_Cleaned.csv,
produced by nlp.py) so that preprocessing happens only once and both models
work from exactly the same cleaned text.

Dataset

The IMDB Movie Reviews dataset contains 50,000 reviews, split evenly into
25,000 positive and 25,000 negative examples. The dataset is therefore
perfectly balanced (50% / 50%), which means accuracy is a meaningful
evaluation metric and no resampling or class weighting was necessary.

After preprocessing, a small number of reviews could become empty strings
(for example, very short reviews consisting only of stop words). These empty
rows were dropped before training to avoid feeding invalid samples into the
vectorizer.

Method

Text representation: TF-IDF

The cleaned reviews were converted into numerical features using
TF-IDF (Term Frequency – Inverse Document Frequency). TF-IDF weights each
term by how often it appears in a review, while down-weighting terms that are
common across the whole corpus. This highlights the words that are most
informative for distinguishing positive from negative reviews.

We included bigrams (word pairs) in addition to single words. This is
important because our preprocessing intentionally keeps negation words
("not", "no", "nor", "never"). Bigrams allow the model to capture phrases such
as "not good" as a single feature, instead of seeing only the isolated words
"not" and "good".

Classifier: Multinomial Naive Bayes

Multinomial Naive Bayes is well suited to text classification with count- or
frequency-based features such as TF-IDF. It is computationally cheap, trains in
seconds even on 40,000 reviews, and provides a strong, interpretable baseline.

Train / test split

The data was split 80% training / 20% test using
train_test_split(test_size=0.2, random_state=42, stratify=y).


stratify=y keeps the 50/50 class balance in both the training and test sets.
random_state=42 makes the split reproducible. The LSTM model uses the same
split with the same random seed, so both models are evaluated on exactly the
same 10,000 test reviews. This is what makes the final comparison fair.


The test set was kept completely untouched during hyperparameter tuning, so the
reported test accuracy is an honest estimate of generalization.

Hyperparameter Tuning

We tuned the model with GridSearchCV using 5-fold stratified
cross-validation on the training set only. GridSearchCV exhaustively tries
every combination of the parameters below, evaluates each with cross-validation,
and keeps the combination with the best mean validation accuracy.

A Pipeline (TF-IDF → Naive Bayes) was used so that the vectorizer is
re-fitted on each training fold separately. This prevents data leakage from the
validation folds into the TF-IDF statistics.

ParameterValues triedMeaningtfidf__ngram_range(1,1), (1,2)unigrams only vs. unigrams + bigramstfidf__min_df1, 5ignore terms appearing in fewer than N reviewstfidf__max_features20000, 40000maximum vocabulary sizenb__alpha0.1, 0.5, 1.0Laplace/Lidstone smoothing strength

alpha is the most important Naive Bayes hyperparameter: a small alpha
trusts the training counts more (risk of overfitting to rare words), while a
large alpha smooths more (risk of underfitting).

Selected configuration

GridSearchCV selected the following best parameters:

ParameterBest valuenb__alpha0.1tfidf__ngram_range(1, 2)tfidf__max_features40000tfidf__min_df1

Best cross-validation accuracy: 0.8805

The fact that bigrams (1, 2) and the larger vocabulary 40000 were selected
confirms that word pairs and a richer vocabulary genuinely help on this task —
consistent with our decision to preserve negation words during preprocessing.

Results

On the untouched 10,000-review test set, the tuned model achieved:

Test accuracy: 0.8832 (88.32%)

ClassPrecisionRecallF1-scoreSupportnegative (0)0.890.870.885000positive (1)0.880.890.885000accuracy0.8810000

The scores are almost identical for both classes, which is expected given the
balanced dataset and shows the model has no strong bias toward either sentiment.

Confusion matrix

The normalized confusion matrix (each row sums to 1):

predicted negativepredicted positivetrue negative0.8740.126true positive0.1080.892

The model correctly identifies 87.4% of negative reviews and 89.2% of
positive reviews. Errors are roughly symmetric, so there is no systematic
tendency to over-predict one class. (See confusion_matrix_nb.png.)

Most informative words

To interpret what the model learned, we plotted the words that most strongly
push a prediction toward positive vs. negative, based on the difference in
log-probabilities log P(word|positive) − log P(word|negative)
(see top_features_nb.png). The strongest positive and negative words match
human intuition (e.g. praise words drive the positive class, criticism words
drive the negative class), which is a useful sanity check that the model relies
on meaningful features.

Problems Encountered and Solutions


Empty reviews after preprocessing. Some reviews became empty strings once
HTML, punctuation, numbers and stop words were removed. These were dropped
before training to prevent the vectorizer from producing invalid (NaN)
features.
Avoiding data leakage during tuning. Fitting TF-IDF on the full dataset
before cross-validation would leak information from the validation folds into
the features. We solved this by wrapping the vectorizer and classifier in a
Pipeline, so TF-IDF is re-fitted on each fold's training portion only.
Negation handling. Sentiment depends heavily on negation ("not good" is
negative). Single words alone lose this. Keeping negation words in
preprocessing and allowing bigrams in TF-IDF together let the model
capture these phrases.
Tuning runtime. GridSearchCV trains 120 models (24 combinations × 5
folds), which takes a few minutes. We used n_jobs=-1 to run folds in
parallel across all CPU cores.


Real-World Relevance

Sentiment classification of this kind is widely used in practice: analyzing
product and movie reviews, monitoring brand perception on social media,
prioritizing negative customer feedback for support teams, and aggregating
opinion trends from large volumes of text. A lightweight, interpretable model
like Naive Bayes is often valuable in production precisely because it is fast,
cheap to run, and easy to explain — making it a sensible baseline to compare a
heavier deep learning model against.

How to Run

python naive_bayes.py

The script prints the dataset statistics, the best hyperparameters, the test
accuracy and classification report, and saves two figures:
confusion_matrix_nb.png and top_features_nb.png.

IMDB_Cleaned.csv must be in the same folder (it is produced by running
nlp.py once beforehand).