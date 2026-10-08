# Movie Review Sentiment Classifier

A bidirectional LSTM with attention that classifies IMDB movie reviews as positive or negative. After training, you can type your own reviews and see what it thinks.

## How it works

- **Data:** the IMDB dataset (25,000 training and 25,000 test reviews) from Hugging Face `datasets`. 10 % of the training set is held out for validation.
- **Tokenising:** reviews are split into word pieces with the `bert-base-uncased` tokenizer and cut to 300 tokens. Only the tokenizer is borrowed; the model itself is trained from scratch.
- **Augmentation:** each training review has about 10 % of its words randomly deleted, so the model can't lean on any single word.
- **Model:** an embedding layer feeds a 2-layer bidirectional LSTM. An attention layer then weighs each word's importance, and a small classifier with dropout produces the final positive/negative score.
- **Training:** Adam, batch size 64, up to 20 epochs. Training stops early when validation accuracy hasn't improved for 2 epochs. The best checkpoint is then scored once on the test set.

## Run it

```bash
pip install -r requirements.txt
python sentiment.py
```

The script downloads the dataset and tokenizer and trains the model. It prints the validation accuracy each epoch and the test accuracy at the end. It then lets you type reviews to classify; type `exit` to quit. A GPU is strongly recommended.

## What I learned

- **Most of programming is debugging.** My first version only reached 49 %, no better than guessing, because I had limited training too tightly. Data augmentation, a bidirectional LSTM and early stopping improved it a lot.
- **Keep the test set separate.** An earlier version used the test set to decide when to stop training and which checkpoint to keep. That makes the reported test accuracy optimistic, because the test set has influenced the model. The current version makes those decisions on a separate validation set and touches the test set only once, at the end.
- **Check what the model already outputs.** The model ends in a sigmoid, and my prediction function applied a second one. That squashed every confidence into the 50–73 % range. Removing it fixed the confidence scores.
- **Real inputs differ from training data.** The model did noticeably worse on my own short reviews than on IMDB's long ones. Good accuracy on a benchmark doesn't guarantee good accuracy on new kinds of text.

I debugged this with help from ChatGPT.
