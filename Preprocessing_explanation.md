Text Preprocessing

Before training the sentiment classification model, the IMDB reviews were preprocessed to reduce noise and standardize the text. Several preprocessing steps were applied to improve the quality of the input data and help the model focus on meaningful information.

First, HTML tags such as <br />, which are commonly found in the IMDB dataset, were removed using BeautifulSoup. These tags do not contribute any useful semantic information and would otherwise be treated as irrelevant tokens.

Next, all text was converted to lowercase. This ensures that words such as "Movie", "movie", and "MOVIE" are treated as the same term, reducing vocabulary size and improving consistency.

Common English contractions were then expanded (for example, "don't" became "do not" and "isn't" became "is not"). This step is particularly important for sentiment analysis because contractions often contain negations that influence the meaning of a sentence.

Any URLs were removed because web links generally do not provide useful information for determining whether a review expresses a positive or negative sentiment.

Punctuation marks, numbers, and other non-alphabetic characters were removed using regular expressions. This helps eliminate noise and leaves only the textual content relevant for sentiment prediction.

After cleaning, the text was tokenized by splitting it into individual words. Stop words were then removed using the NLTK English stop word list. However, the negation words "not", "no", "nor", and "never" were intentionally retained because they carry important sentiment information. For example, removing "not" from the phrase "not good" would incorrectly change its meaning to "good".

Finally, lemmatization was applied using WordNetLemmatizer. Lemmatization reduces words to their base form while preserving their meaning. For example, "movies" becomes "movie" and "running" becomes "run". Lemmatization was chosen instead of stemming because it produces valid dictionary words and generally preserves semantic meaning better, which is beneficial for sentiment analysis tasks.

The cleaned words were then combined back into a single string and stored in the dataset. Additionally, the sentiment labels were encoded numerically, with positive reviews represented as 1 and negative reviews represented as 0, making them suitable for machine learning algorithms.

A key design choice was preserving negation words and using lemmatization rather than aggressive stemming, since sentiment classification depends heavily on subtle differences in meaning. This approach reduces noise while retaining information that helps distinguish positive and negative reviews.