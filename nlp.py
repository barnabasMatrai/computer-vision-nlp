
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


# -------------------------------------------------------
# Split data into training and test sets
# -------------------------------------------------------

from sklearn.model_selection import train_test_split

# X is the review texts, y is the labels (0 or 1)
X = df["review"]
y = df["label"]

# 80% training, 20% test
# stratify=y keeps the same ratio of positive/negative in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training reviews:", len(X_train))
print("Test reviews:    ", len(X_test))


import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# -------------------------------------------------------
# LSTM Model using PyTorch
# -------------------------------------------------------

# ---- Define hyperparameters to try ---

# Each key holds a list of values we want to test.
# At the end we will loop over all combinations and compare results.
hyperparameter_options = {
    "vocab_size":    [10000, 20000],
    "max_length":    [100, 200],
    "embedding_dim": [64, 128],
    "hidden_size":   [64, 128],
    "num_layers":    [1, 2],
    "learning_rate": [0.001, 0.0005],
    "batch_size":    [32, 64],
    "dropout":       [0.2, 0.5]
}

print("Hyperparameter options defined.")

# We use random search instead of grid search.
# Grid search tries every single combination (2^7 = 128 combinations here).
#128 full model trainings which would take hours.
# Random search picks a small random subset of combinations instead.

NUM_RANDOM_COMBINATIONS = 5

all_combinations = []

for vocab_size in hyperparameter_options["vocab_size"]:
    for max_length in hyperparameter_options["max_length"]:
        for embedding_dim in hyperparameter_options["embedding_dim"]:
            for hidden_size in hyperparameter_options["hidden_size"]:
                for num_layers in hyperparameter_options["num_layers"]:
                    for learning_rate in hyperparameter_options["learning_rate"]:
                        for batch_size in hyperparameter_options["batch_size"]:
                            combination = {
                                "vocab_size":    vocab_size,
                                "max_length":    max_length,
                                "embedding_dim": embedding_dim,
                                "hidden_size":   hidden_size,
                                "num_layers":    num_layers,
                                "learning_rate": learning_rate,
                                "batch_size":    batch_size
                            }
                            all_combinations.append(combination)

# Randomly pick 5 combinations to try
random.seed(42)
selected_combinations = random.sample(all_combinations, NUM_RANDOM_COMBINATIONS)
print("Total possible combinations:", len(all_combinations))
print("Combinations we will try:", NUM_RANDOM_COMBINATIONS)


# ---- LSTM ----

# ---- Create PyTorch Dataset ----

class SentimentDataset(Dataset):
    # PyTorch needs data in a special Dataset format to load it in batches.
    # We create our own Dataset class by extending PyTorch's Dataset class.

    def __init__(self, sequences, labels):
        # Convert Python lists to PyTorch tensors
        # torch.long = integer numbers (for word indexes)
        # torch.float32 = decimal numbers (for labels 0 and 1)
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels    = torch.tensor(labels,    dtype=torch.float32)

    def __len__(self):
        # Returns how many reviews are in this dataset
        return len(self.labels)

    def __getitem__(self, index):
        # Returns one review and its label at the given position
        one_sequence = self.sequences[index]
        one_label    = self.labels[index]
        return one_sequence, one_label
    

# ---- LSTM Model ----

class SentimentLSTM(nn.Module):
    # Our model has 3 main parts:
    # 1. Embedding layer  → turns word indexes into meaningful vectors
    # 2. LSTM layer       → reads the vectors in order and remembers patterns
    # 3. Output layer     → converts the final LSTM memory into 0 or 1

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super(SentimentLSTM, self).__init__()

        # Embedding: each word index becomes a vector of size embedding_dim
        # padding_idx=0 means the <PAD> token always gives a zero vector (no information)
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embedding_dim,
            padding_idx    = 0
        )

        # LSTM: reads the word vectors one by one and updates its memory
        # batch_first=True means input shape is (batch_size, sequence_length, embedding_dim)
        # dropout is applied between LSTM layers to reduce overfitting
        self.lstm = nn.LSTM(
            input_size  = embedding_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0  # no dropout if there is only 1 layer
        )

        # Dropout: randomly turns off some neurons during training
        # this forces the model to not rely too much on any single neuron
        self.dropout = nn.Dropout(dropout)

        # Output layer: takes the final LSTM memory and gives one number
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):

        # Step 1: convert word indexes to word vectors
        embedded = self.embedding(x)

        # Step 2: pass word vectors through the LSTM
        # lstm_output and cell are not used, we only need the final hidden state
        lstm_output, (hidden, cell) = self.lstm(embedded)

        # Step 3: take only the last layer's final hidden state
        last_hidden = hidden[-1]

        # Step 4: apply dropout
        last_hidden = self.dropout(last_hidden)

        # Step 5: pass through output layer and remove extra dimension
        output = self.output_layer(last_hidden)
        output = output.squeeze(1)

        return output
    

# ---- Helper functions for the tuning loop ----

def build_vocabulary(X_train, vocab_size):

    # Count how many times each word appears in training reviews
    word_counts = Counter()
    for review in X_train:
        words = review.split()
        word_counts.update(words)

    # Take the most common words (minus 2 for <PAD> and <UNK>)
    most_common_words = word_counts.most_common(vocab_size - 2)

    # Create word → number dictionary
    # <PAD> = 0, <UNK> = 1, then all other words start from 2
    word_to_index = {"<PAD>": 0, "<UNK>": 1}
    for idx, (word, count) in enumerate(most_common_words, start=2):
        word_to_index[word] = idx

    return word_to_index

def convert_to_sequences(X_data, word_to_index, max_length):

    all_sequences = []

    for review in X_data:
        words    = review.split()
        sequence = []

        # Replace each word with its number from the vocabulary
        for word in words:
            if word in word_to_index:
                sequence.append(word_to_index[word])
            else:
                sequence.append(1)  # 1 = <UNK>, word not in vocabulary

        # If the review is too long, cut it from the end
        if len(sequence) > max_length:
            sequence = sequence[:max_length]

        # If the review is too short, add zeros at the beginning
        while len(sequence) < max_length:
            sequence = [0] + sequence  # 0 = <PAD>

        all_sequences.append(sequence)

    return all_sequences


def create_dataloaders(X_train_sequences, X_test_sequences, y_train_list, y_test_list, batch_size):

    # Create Dataset objects
    train_dataset = SentimentDataset(X_train_sequences, y_train_list)
    test_dataset  = SentimentDataset(X_test_sequences,  y_test_list)

    # Create DataLoader objects
    # shuffle=True for training so the model doesn't memorize the order
    # shuffle=False for testing so results are consistent
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_one_epoch(model, train_loader, criterion, optimizer, device):

    # Tell PyTorch we are in training mode (activates dropout)
    model.train()

    total_loss      = 0
    correct         = 0
    total_samples   = 0

    for batch_sequences, batch_labels in train_loader:

        # Move data to the same device as the model (GPU or CPU)
        batch_sequences = batch_sequences.to(device)
        batch_labels    = batch_labels.to(device)

        # Forward pass: calculate predictions
        predictions = model(batch_sequences)

        # Calculate how wrong the predictions are
        loss = criterion(predictions, batch_labels)

        # Backward pass: update model weights
        optimizer.zero_grad()  # clear old gradients
        loss.backward()        # calculate new gradients
        optimizer.step()       # update weights

        # Track loss and correct predictions
        total_loss    = total_loss + loss.item()
        predicted_labels = (predictions >= 0.0).float()
        correct       = correct + (predicted_labels == batch_labels).sum().item()
        total_samples = total_samples + len(batch_labels)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total_samples

    return avg_loss, accuracy

def evaluate_model(model, test_loader, criterion, device):

    # Tell PyTorch we are in evaluation mode (deactivates dropout)
    model.eval()

    total_loss    = 0
    correct       = 0
    total_samples = 0

    # torch.no_grad() means we don't calculate gradients
    # we don't need them during evaluation, so this saves memory and time
    with torch.no_grad():

        for batch_sequences, batch_labels in test_loader:

            batch_sequences = batch_sequences.to(device)
            batch_labels    = batch_labels.to(device)

            predictions = model(batch_sequences)
            loss        = criterion(predictions, batch_labels)

            total_loss       = total_loss + loss.item()
            predicted_labels = (predictions >= 0.0).float()
            correct          = correct + (predicted_labels == batch_labels).sum().item()
            total_samples    = total_samples + len(batch_labels)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total_samples

    return avg_loss, accuracy


def run_training(model, train_loader, test_loader, criterion, optimizer, device):

    NUM_EPOCHS = 50     # large number, early stopping will handle when to stop
    PATIENCE   = 3      # stop if no improvement for 3 epochs in a row
    MIN_DELTA  = 0.001  # minimum improvement to count as "getting better"

    # Scheduler reduces learning rate when validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode    = "min",  # we want loss to go DOWN
        factor  = 0.5,    # cut learning rate in half when plateau is detected
        patience= 2       # wait 2 epochs before reducing
    )

    best_val_accuracy          = 0.0
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):

        # Train for one epoch
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate on test set
        val_loss, val_accuracy = evaluate_model(
            model, test_loader, criterion, device
        )

        print(
            "  Epoch", epoch + 1, "/", NUM_EPOCHS,
            " | Train Loss:", round(train_loss, 4),
            " Train Acc:",    round(train_accuracy * 100, 2), "%",
            " | Val Loss:",   round(val_loss, 4),
            " Val Acc:",      round(val_accuracy * 100, 2), "%"
        )

        # Tell the scheduler what the validation loss was this epoch
        # if it does not improve for 2 epochs, learning rate will be cut in half
        scheduler.step(val_loss)

        # Early stopping check
        # we only count it as improvement if accuracy went up by at least MIN_DELTA
        if val_accuracy > best_val_accuracy + MIN_DELTA:
            best_val_accuracy          = val_accuracy
            epochs_without_improvement = 0
            # Save the best model weights so we can restore them later
            torch.save(model.state_dict(), "best_lstm_model.pt")
        else:
            epochs_without_improvement = epochs_without_improvement + 1
            if epochs_without_improvement >= PATIENCE:
                print("  Early stopping triggered.")
                break

    return best_val_accuracy


# ---- Main Hyperparameter Tuning Loop ----

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
y_train_list = y_train.tolist()
y_test_list  = y_test.tolist()

all_tuning_results = []

print("Using device:", device)
print("Starting hyperparameter tuning...")
print()

combination_number = 0

for combination in selected_combinations:

    combination_number = combination_number + 1

    vocab_size    = combination["vocab_size"]
    max_length    = combination["max_length"]
    embedding_dim = combination["embedding_dim"]
    hidden_size   = combination["hidden_size"]
    num_layers    = combination["num_layers"]
    learning_rate = combination["learning_rate"]
    batch_size    = combination["batch_size"]
    dropout       = combination["dropout"]

    print("------------------------------------------------------------")
    print("Combination", combination_number + 1, "/", NUM_RANDOM_COMBINATIONS)
    print("  vocab_size:   ", vocab_size)
    print("  max_length:   ", max_length)
    print("  embedding_dim:", embedding_dim)
    print("  hidden_size:  ", hidden_size)
    print("  num_layers:   ", num_layers)
    print("  learning_rate:", learning_rate)
    print("  batch_size:   ", batch_size)
    print("  dropout:      ", dropout)
    print()

    # Step 1: build vocabulary
    word_to_index = build_vocabulary(X_train, vocab_size)

    # Step 2: convert reviews to sequences
    X_train_sequences = convert_to_sequences(X_train, word_to_index, max_length)
    X_test_sequences  = convert_to_sequences(X_test,  word_to_index, max_length)

    # Step 3: create dataloaders
    train_loader, test_loader = create_dataloaders(
        X_train_sequences, X_test_sequences,
        y_train_list, y_test_list,
        batch_size
    )

    # Step 4: create model
    model = SentimentLSTM(
        vocab_size    = vocab_size,
        embedding_dim = embedding_dim,
        hidden_size   = hidden_size,
        num_layers    = num_layers,
        dropout       = dropout
    )
    model = model.to(device)

    # Step 5: set up loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Step 6: train the model
    best_accuracy = run_training(
        model, train_loader, test_loader, criterion, optimizer, device
    )

    # Save result
    result = {
        "combination": combination,
        "accuracy":    round(best_accuracy * 100, 2)
    }
    all_tuning_results.append(result)

    print("  Best accuracy for this combination:", round(best_accuracy * 100, 2), "%")


# ---- Find and print the best result ----

best_lstm_result = all_tuning_results[0]
for result in all_tuning_results:
    if result["accuracy"] > best_lstm_result["accuracy"]:
        best_lstm_result = result

print()
print("------------------------------------------------------------")
print("HYPERPARAMETER TUNING COMPLETE")
print("------------------------------------------------------------")
print()
print("All LSTM results:")
for result in all_tuning_results:
    print("  Accuracy:", result["accuracy"], "%  |  Params:", result["combination"])

print()
print("Best LSTM combination:")
for param_name, param_value in best_lstm_result["combination"].items():
    print("  ", param_name, ":", param_value)

print()
print("Best LSTM accuracy:", best_lstm_result["accuracy"], "%")
print("------------------------------------------------------------")