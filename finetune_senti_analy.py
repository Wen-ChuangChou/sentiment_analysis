from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from datetime import datetime
# import torch
import json
import os
from datasets import load_dataset
import numpy as np
import evaluate

model_name = "Llama-3.1-8B-Instruct"
model_path = os.path.join("./pretrained_llms", model_name)
data_path = "./data"
data_name = "mteb/tweet_sentiment_extraction"
cache_dir = "./cache"
output_dir="./results"

dataset = load_dataset(data_name, cache_dir=data_path)

# Load accuracy metric
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Compute individual metrics
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    precision = precision_score(labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(labels, predictions, average="weighted")
    f1 = f1_score(labels, predictions, average="weighted")
    
    # Return all metrics
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          add_eos_token=True,
                                          cache_dir=cache_dir)
if tokenizer.pad_token_id is None:
    print("No pad token found, setting pad token to eos token")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id   

def tokenizer_function(examples):
    return tokenizer(examples['text'], truncation=True)   
# apply tokenizer function on your data
tokenized_data = dataset.map(tokenizer_function, batched=True)

train = tokenized_data['train']
test = tokenized_data['test']

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="test_trainer",
    learning_rate=1e-5,  # Experiment with different rates
    # lr_scheduler_type="linear",  # Add learning rate scheduling
    # warmup_steps=100,  # Implement learning rate warmup
    optim="adamw_torch",
    weight_decay=0.01,
    num_train_epochs=10,
    save_strategy='steps',
    save_steps=500,   
    eval_strategy='steps',
    logging_steps=250,
    load_best_model_at_end=True,
    save_total_limit=1,
    # eval_steps=50,
    # gradient_accumulation_steps=4,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
print("\n=== Starting Training ===")
trainer.train()

save_model_path = os.path.join("./saved_model", model_name)
trainer.save_model(save_model_path)
print(f"Fine-tuned model saved to: {save_model_path}")

