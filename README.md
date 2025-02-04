# Fine-Tuning Llama3 for Sentiment Analysis  

Welcome to the repository for fine-tuning the **Llama3** model for sentiment analysis. This project aims to enhance **Llama3**, a state-of-the-art language model, to accurately classify sentiments in text data.

## Overview  
Sentiment analysis is a key task in **natural language processing (NLP)** that involves determining the sentiment expressed in a piece of text. By fine-tuning **Llama3**, we aim to improve its ability to classify text as **positive, negative, or neutral**.

## Data  
The dataset used for fine-tuning and testing **Llama3** comes from the **Massive Text Embedding Benchmark (MTEB)**. It consists of tweets labeled with one of three sentiments: **positive, neutral, or negative**.  

- The dataset can be accessed from Hugging Face under the name **"mteb/tweet_sentiment_extraction"**.  
- It contains **27,481** sentence pairs in the training set and **3,534** in the test set.  

## Results  
We first evaluated the **Llama-3.1-8B Instruct** foundation model on sentiment classification without fine-tuning. Using the following prompt:  

> *"Analyze the sentiment of the following text. Respond with exactly one word: either 'positive', 'negative', or 'neutral'."*  

The model achieved an **accuracy of 63.41%** when its predictions were compared with the labeled data.  

After fine-tuning the model on the training set, the accuracy improved significantly to **81.49%**.  