LSTM Sentiment Classifier

Overview

For this project we built an LSTM model to classify IMDB movie reviews as positive or negative. LSTM stands for Long Short-Term Memory. It is a type of neural network that reads text word by word and keeps a kind of "memory" as it goes, so it can understand how earlier words in a sentence affect the meaning of later words. We chose LSTM because our other model, Naive Bayes, treats every word independently. LSTM can learn things like "not bad" being positive, because it sees the words together in order.


From Text to Numbers

Neural networks need numbers as input, not words. So the first thing we had to do was convert each review into a sequence of numbers.

First we built a vocabulary from the training data. We counted how often each word appeared and kept only the most common ones. We did not use all words because rare words (like typos or very specific names) usually do not help the model learn general patterns, they just add noise.

We kept two special slots in the vocabulary. Index 0 is for PAD and index 1 is for UNK. PAD is used to fill up short reviews so every sequence has the same length. UNK is used when a word from the test set was not seen during training.

After building the vocabulary, we replaced each word in every review with its index number. If a review was longer than our max length we cut it. If it was shorter we added PAD tokens at the beginning. We pad at the beginning instead of the end because the LSTM reads left to right and we want the actual words to be as close to the output as possible.


Model Architecture

The model has three parts.

The first part is the Embedding layer. This takes each word index and converts it into a small vector of numbers. These vectors are learned during training. Over time the model adjusts them so that words with similar meanings end up with similar vectors. We set padding_idx=0 so the PAD token always gives a zero vector and does not affect learning.

The second part is the LSTM layer itself. It reads the word vectors one by one and updates its internal memory at each step. At the end it outputs a final hidden state which is basically a summary of the whole review. We had the option to stack multiple LSTM layers on top of each other (num_layers). When we used more than one layer we applied dropout between them to prevent overfitting.

We also initialized the forget gate bias to 1. By default the LSTM tends to forget too much information in the early epochs. Setting this bias to 1 makes it remember more at the start, which leads to more stable training.

The third part is a simple linear output layer. It takes the final hidden state and produces one number. If that number is positive the model predicts a positive review, if it is negative the model predicts a negative review. We used BCEWithLogitsLoss as the loss function because it handles the conversion to a probability internally and is more numerically stable.


Training

We used a few techniques to make training more stable and to avoid overfitting.

Gradient clipping: sometimes during training the gradients can get very large and cause the model weights to jump to extreme values. We clipped gradients to a max norm of 1.0 to prevent this.

Learning rate scheduler: we used ReduceLROnPlateau. If the test loss does not improve for 2 epochs, the learning rate is automatically cut in half. This way the model makes smaller and more careful updates as it gets closer to a good solution.

Early stopping: we stopped training if the test accuracy did not improve by at least 0.1% for 3 epochs in a row. This saves time and prevents the model from overfitting to the training data. Whenever a new best accuracy was found, we saved a copy of the model weights in memory so we could go back to the best version at the end.


Hyperparameter Tuning

We could not use grid search like we did for Naive Bayes because training one LSTM takes several minutes. Grid search would mean training hundreds of models which would take way too long. So we used random search instead. We randomly picked 5 combinations of hyperparameters and trained one model for each combination.

The things we tuned were: vocabulary size, maximum sequence length, embedding dimension, LSTM hidden size, number of LSTM layers, batch size and dropout rate. The learning rate was kept fixed at 0.001 because the scheduler already adjusts it during training automatically.

After all 5 combinations finished training we compared their best test accuracies and kept the weights of the best one in memory.

One thing we had to be careful about: in an earlier version of the code, every combination was saving its best weights to the same file. This meant the file always ended up with the last combination's weights, not the best combination's weights. We fixed this by tracking a global best accuracy across all combinations and only updating the saved weights when a new overall best was reached.


Things We Could Have Done Differently

For the embeddings, we trained our own word vectors from scratch. A common alternative is to use pretrained embeddings like GloVe or Word2Vec, which were already trained on billions of words and know that words like "good" and "great" are similar. We did not use them because we wanted to keep everything self-contained and train purely on the IMDB data.

For the architecture, we only used the final hidden state of the LSTM as the summary of the review. A Bidirectional LSTM would read the text both forward and backward and combine both results. This can help because some words get more context from what comes after them. We left it out to keep the model simpler and training faster.

We also could have added an attention mechanism. Instead of just using the last hidden state, attention lets the model look back at all the hidden states and decide which words mattered most for the prediction. This usually improves results but makes the code more complex, so we did not include it.

For hyperparameter tuning, Bayesian optimization would have been smarter than random search because it learns from previous results and focuses on more promising areas. We used random search because it is much simpler to implement and still works well enough for a small number of combinations.

Another option instead of LSTM was GRU (Gated Recurrent Unit). GRU is similar to LSTM but has fewer parameters and trains faster. In practice GRU and LSTM often give similar results on text classification tasks. We chose LSTM because it is the more commonly used and well-known architecture.
