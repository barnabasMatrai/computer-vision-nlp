"""
Naive Bayes Sentiment Classifier

Uses TF-IDF features and a Multinomial Naive Bayes classifier
for sentiment classification on the IMDB dataset.
Hyperparameter tuning is performed using GridSearchCV.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Reproducibility
RANDOM_STATE = 42

# 1. Load the cleaned dataset
df = pd.read_csv("IMDB_Cleaned.csv")

# Remove missing or empty reviews
df = df.dropna(subset=["review"])
df = df[df["review"].str.strip() != ""]

print("Dataset shape:", df.shape)
print("\nClass distribution (label):")
print(df["label"].value_counts())
print("\nClass balance (%):")
print((df["label"].value_counts(normalize=True) * 100).round(2))

X = df["review"]
y = df["label"]

# 2. Train / test split
# Split data while keeping class balance.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nTrain samples: {len(X_train)}  |  Test samples: {len(X_test)}")

# 3. TF-IDF + Naive Bayes pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB()),
])

# HYPERPARAMETER TUNING
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 5],
    "tfidf__max_features": [20000, 40000],
    "nb__alpha": [0.1, 0.5, 1.0],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,                 # 5-fold cross-validation
    scoring="accuracy",
    n_jobs=-1,            # use all CPU cores
    verbose=1,
)

print("\nRunning GridSearchCV (this can take a few minutes)...")
grid_search.fit(X_train, y_train)

print("\n" + "=" * 50)
print("BEST HYPERPARAMETERS FOUND")
print("=" * 50)
for param, value in grid_search.best_params_.items():
    print(f"  {param:25s} : {value}")
print(f"\nBest cross-validation accuracy: {grid_search.best_score_:.4f}")

# The best model (already refitted on the full training set by GridSearchCV)
best_model = grid_search.best_estimator_


# 4. Evaluate the tuned model on the untouched test set
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'=' * 50}")
print(f"TEST ACCURACY (tuned model): {accuracy:.4f}")
print(f"{'=' * 50}\n")

print("Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["negative (0)", "positive (1)"],
))

# 5. Confusion matrix (heatmap)
cm = confusion_matrix(y_test, y_pred)

# Visualize confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["negative", "positive"],
    yticklabels=["negative", "positive"],
    cbar=False,
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title(f"Naive Bayes (tuned) - Confusion Matrix\nAccuracy: {accuracy:.4f}")
plt.tight_layout() # Save and display figure
plt.savefig("confusion_matrix_nb.png", dpi=150)
print("Saved: confusion_matrix_nb.png")
plt.show()

# Normalized confusion matrix (percentages)
cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
print("\nNormalized confusion matrix (row = true class):")
print(pd.DataFrame(
    cm_norm.round(3),
    index=["true negative", "true positive"],
    columns=["pred negative", "pred positive"],
))

# 6. Most informative words per class
# Identify the strongest positive and negative sentiment words.
# Extract the best TF-IDF vectorizer and Naive Bayes model
best_vectorizer = best_model.named_steps["tfidf"]
best_nb = best_model.named_steps["nb"]

# Get feature names and calculate sentiment importance scores
feature_names = np.array(best_vectorizer.get_feature_names_out())
log_prob_diff = best_nb.feature_log_prob_[1] - best_nb.feature_log_prob_[0]

# Select top positive and negative words
top_n = 15
top_positive_idx = np.argsort(log_prob_diff)[-top_n:]
top_negative_idx = np.argsort(log_prob_diff)[:top_n]

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Positive sentiment words
axes[0].barh(
    feature_names[top_positive_idx],
    log_prob_diff[top_positive_idx],
    color="seagreen",
)
axes[0].set_title("Top words -> POSITIVE")
axes[0].set_xlabel("log P(word|pos) - log P(word|neg)")

# Negative sentiment words
axes[1].barh(
    feature_names[top_negative_idx],
    log_prob_diff[top_negative_idx],
    color="indianred",
)
axes[1].set_title("Top words -> NEGATIVE")
axes[1].set_xlabel("log P(word|pos) - log P(word|neg)")

# Save and display figure
plt.tight_layout()
plt.savefig("top_features_nb.png", dpi=150)
print("Saved: top_features_nb.png")
plt.show()

print("\nDone.")

