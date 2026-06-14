import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

# Initialize tools
lemmatizer = WordNetLemmatizer()

# Keep negation words because they're important for sentiment
stop_words = set(stopwords.words('english'))
stop_words = stop_words - {'not', 'no', 'nor', 'never'}

# Common contractions
contractions = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "can't": "cannot",
    "couldn't": "could not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "mustn't": "must not"
}

def expand_contractions(text):
    for contraction, expanded in contractions.items():
        text = re.sub(
            r'\b' + re.escape(contraction) + r'\b',
            expanded,
            text,
            flags=re.IGNORECASE
        )
    return text

def preprocess_text(text):
    
    # Remove HTML tags like <br />
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

    # Lowercase
    text = text.lower()

    # Expand contractions
    text = expand_contractions(text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Apply preprocessing
#df["clean_review"] = df["review"].apply(preprocess_text)
df["review"] = df["review"].apply(preprocess_text)

# Encode labels
df["label"] = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})

#print(df[["review", "clean_review", "sentiment", "label"]].head())
print(df[["review", "sentiment", "label"]].head())

# save cleaned dataset
df.to_csv("IMDB_Cleaned.csv", index=False)
