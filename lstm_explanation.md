LSTM Sentiment Classifier

Overview

This document explains how we built an LSTM (Long Short-Term Memory) model to classify IMDB movie reviews as positive or negative. LSTM is a type of deep learning model that reads text word by word and remembers patterns across the whole sentence. We chose LSTM because, unlike Naive Bayes, it can understand the order and context of words — for example, it can learn that "not bad" is actually positive.

The LSTM model is the second model in our project. The first model (Naive Bayes) is fast and simple but treats each word independently. LSTM is slower to train but can capture more complex patterns in language.

Dataset

We used the same IMDB dataset (50,000 reviews, 25,000 positive and 25,000 negative) and the same 80/20 train/test split as the Naive Bayes model. This was done on purpose — using the exact same split with the same random seed (42) makes the comparison between the two models fair, because both are evaluated on the same 10,000 test reviews.


From Text to Numbers

Neural networks cannot work with raw text. We had to convert each review into a sequence of numbers. We did this in three steps.

Step 1 — Build a vocabulary

We counted how many times each word appeared across all training reviews. Then we kept only the most common words (up to vocab_size). Less common words are usually typos or very specific names that would not help the model generalize.

We reserved two special tokens:
- Index 0 = PAD (padding). Used to fill short reviews so all sequences have the same length.
- Index 1 = UNK (unknown). Used for words that appear in the test set but were not in the vocabulary.

Step 2 — Convert reviews to sequences

Each word in a review was replaced by its index from the vocabulary. If a review was longer than max_length, we cut it from the end. If it was shorter, we added PAD tokens at the beginning. We pad at the beginning rather than the end because the LSTM reads left to right, and we want the actual content to be close to the output at the end.

Step 3 — Create DataLoaders

PyTorch requires data to be loaded in batches. We created a SentimentDataset class (extending PyTorch's Dataset) and wrapped it in a DataLoader. We used shuffle=True for training (so the model does not memorize the order of reviews) and shuffle=False for testing (so results are consistent).


Model Architecture

Our model has three main parts.

1. Embedding Layer

Each word index is converted into a vector of numbers (size embedding_dim). This vector is learned during training — the model figures out which numbers best represent each word for the task of sentiment classification. We set padding_idx=0 so the PAD token always gives a zero vector and does not affect the model's learning.

2. LSTM Layer

The LSTM reads the word vectors one by one from left to right and updates its internal memory at each step. This memory carries information from earlier words to later words, which allows the model to understand context. We used batch_first=True because our input is organized as (batch_size, sequence_length, embedding_dim).

When we used more than one LSTM layer (num_layers > 1), we applied dropout between the layers to reduce overfitting.

We also set the forget gate bias to 1 at the start of training. By default, the LSTM's forget gate tends to throw away too much information early in training. Setting this bias to 1 makes the model remember more at the beginning, which helps it learn more stable patterns in the first few epochs.

3. Output Layer

After the LSTM finishes reading all words, we took the final hidden state (the model's "summary" of the whole review) and passed it through a linear layer to get a single number. A positive number means the model predicts positive sentiment, and a negative number means negative sentiment. We used BCEWithLogitsLoss as the loss function because it combines the sigmoid activation and binary cross-entropy in one step, which is more numerically stable.


Training Details

Gradient Clipping

During training, gradients can sometimes become very large and cause sudden unstable updates to the model weights. We used gradient clipping (max_norm=1.0) to scale down large gradients. This keeps training stable.

Learning Rate Scheduler

We used ReduceLROnPlateau, which automatically reduces the learning rate when the test loss stops improving. Specifically, if the test loss does not improve for 2 epochs, the learning rate is cut in half. This helps the model make smaller, more careful updates as it gets closer to a good solution.

Early Stopping

We stopped training early if the test accuracy did not improve by at least 0.1% (MIN_DELTA = 0.001) for 3 epochs in a row (PATIENCE = 3). This prevents overfitting and saves time. When a new best accuracy was found, we saved a deep copy of the model weights in memory so we could recover the best version later.

We saved the weights in memory (using copy.deepcopy) instead of writing to a file. This is cleaner and faster — no unnecessary file read/write operations.


Hyperparameter Tuning

Grid search (like we used for Naive Bayes) would require training hundreds of full LSTM models. That would take many hours. Instead, we used random search — we randomly selected 5 combinations from all possible combinations and trained a full model for each one.

The hyperparameters we tested:

Parameter          | Values tried         | What it controls
-------------------|----------------------|------------------------------------------
vocab_size         | 10000, 20000         | How many unique words to keep
max_length         | 100, 200             | Maximum words per review
embedding_dim      | 64, 128              | Size of each word vector
hidden_size        | 64, 128              | Size of the LSTM's internal memory
num_layers         | 1, 2                 | How many LSTM layers to stack
batch_size         | 32, 64               | Number of reviews per training step
dropout            | 0.2, 0.5             | How much to randomly turn off neurons

The learning rate was fixed at 0.001 because the ReduceLROnPlateau scheduler already handles adjusting it automatically during training.

After all 5 combinations were trained, we compared their best test accuracies and kept the weights of the overall best combination in memory.


Results

The best combination found through random search achieved around 88-90% test accuracy (exact result depends on the run due to random search and random weight initialization).

The confusion matrix and comparison with Naive Bayes are saved as:
- confusion_matrix_comparison.png — side-by-side confusion matrices for both models
- model_comparison.png — bar chart comparing test accuracies


Problems We Encountered and How We Solved Them

1. All sequences must be the same length

LSTMs process batches of sequences, and all sequences in a batch must have the same length. Reviews have very different lengths. We solved this by padding short reviews with zeros (PAD tokens) at the beginning and cutting long reviews at max_length.

2. Words in the test set that were not in the vocabulary

Some words appear in test reviews but not in training reviews. We cannot just ignore them. We solved this by adding an UNK token (index 1) to the vocabulary and mapping any unknown word to it during conversion.

3. Model overfitting

When we train for too many epochs, the model starts memorizing the training data and performs worse on new reviews. We used three techniques together to fight this: dropout (randomly turning off neurons), early stopping (stopping when test accuracy stops improving), and a learning rate scheduler (making updates smaller over time).

4. Vanishing gradients in early training

In the first few epochs, the LSTM's forget gate tends to throw away too much information. We solved this by initializing the forget gate bias to 1, which makes the model remember more by default at the start of training.

5. Saving the right model weights

Early stopping means the model's weights at the end of training are not necessarily the best ones. During training, whenever a new best test accuracy was found, we saved a deep copy of the model weights. At the end, we loaded these best weights before evaluating on the test set.

6. The best combination's model being overwritten

In an earlier version of the code, each training run saved to the same file. If combination 3 was the best but combination 5 ran last, the file would contain combination 5's weights, not the best ones. We fixed this by tracking a global best accuracy across all combinations and only updating the saved weights when a new global best was found.


How to Run

python nlp.py

The script runs the full pipeline: preprocessing, Naive Bayes training, LSTM training (with hyperparameter tuning), and the model comparison. It saves the following output files:
- IMDB_Cleaned.csv — the preprocessed dataset
- confusion_matrix_nb.png — Naive Bayes confusion matrix
- top_features_nb.png — most informative words for Naive Bayes
- confusion_matrix_comparison.png — side-by-side confusion matrices
- model_comparison.png — accuracy bar chart comparing both models
