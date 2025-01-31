# Fine-Tuning Llama3 for Sentiment Analysis
Welcome to the repository for fine-tuning the Llama3 model specifically for sentiment analysis. This project aims to leverage the capabilities of Llama3, a state-of-the-art language model, to accurately classify sentiments in text data.

## Overview
Sentiment analysis, a crucial task in natural language processing (NLP), involves determining the sentiment expressed in a piece of text. By fine-tuning Llama3, we aim to enhance its performance in classifying text as positive, negative, or neutral.

## Data
The data used for fine-tuning and testing the Llama3 model to recognize emotions in tweets was obtained from the Massive Text Embedding Benchmark (MTEB) database. These tweets are labeled with three different sentiments: positive, neutral, and negative. The database can be downloaded from the Hugging Face datasets under the name "mteb/tweet_sentiment_extraction."
There are 27,481 pairs of sentences in the training set and 3,534 in the test set.

## Results
Accuracy: 63.41% to 80.56%