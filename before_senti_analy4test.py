# Load test data

from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
import pandas as pd
import os
from datetime import datetime
import json


# Fix Random Seeds
seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    print("CUDA is used")
else:
    torch.device("cpu")
    print("CPU is used")

# model_path = "./pretrained_llms/Llama-3.3-70B-Instruct"
# model_path = "./pretrained_llms/Llama-3.1-8B"
model_name = "Llama-3.1-8B-Instruct"
model_path = os.path.join("./pretrained_llms", model_name)
data_path = "./data"
data_name = "mteb/tweet_sentiment_extraction"
cache_dir = "./cache"
output_dir="./results"

dataset = load_dataset(data_name, cache_dir=data_path, split='test')
# dataset = load_dataset(data_name, cache_dir=data_path, split='train')
# dataset = load_dataset(data_name, cache_dir=data_path, split='train', remove_columns=["id"])

def analyze_sentiment_zero_shot(dataset, model, tokenizer, device, verbose=False, num_samples=None, 
                              checkpoint_interval=100, model_name="model", data_name="data", 
                              cache_dir="./cache"):
    """
    Analyze sentiment using zero-shot learning with checkpoint saving and resumption capabilities.
    
    Args:
        dataset: Dataset containing texts to analyze
        model: The language model
        tokenizer: The tokenizer
        device: Device to run the model on
        verbose: If True, prints each text and its prediction
        num_samples: Optional number of samples to process (None for all)
        checkpoint_interval: Number of items to process before saving checkpoint
        model_name: Name of the model for checkpoint filename
        data_name: Name of the dataset for checkpoint filename
        cache_dir: Directory to save checkpoints
    
    Returns:
        list: List of dictionaries containing text and predictions
    """
    model.eval()
    model = model.to(device)
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate checkpoint filename
    dataset_prefix = data_name.split('/')[-1][:5]
    checkpoint_filename = f"checkpoint_test_{model_name}_{dataset_prefix}.json"
    checkpoint_path = os.path.join(cache_dir, checkpoint_filename)
    
    # Initialize or load checkpoint
    start_idx = 0
    results = []
    
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data['results']
                start_idx = checkpoint_data['next_idx']
                print(f"Resuming from checkpoint at index {start_idx}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from beginning...")
    
    prompt_template = """[INST] Analyze the sentiment of the following text. Respond with exactly one word: either 'positive', 'negative', or 'neutral'.

Text: "{}"

Sentiment: [/INST]"""
    
    # Handle num_samples
    texts = dataset['text']
    if num_samples is not None:
        texts = texts[:num_samples]
    
    total = len(texts)
    
    def save_checkpoint(current_idx, current_results):
        checkpoint_data = {
            'next_idx': current_idx + 1,
            'results': current_results,
            'total': total,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=4)
        if verbose:
            print(f"\nCheckpoint saved at index {current_idx}")
    
    try:
        for i in range(start_idx, total):
            text = texts[i]
            prompt = prompt_template.format(text)
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    num_return_sequences=1,
                    temperature=0.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            sentiment = response[len(prompt):].strip().split()[0].lower()
            
            # Store result
            results.append({
                'text': text,
                'predicted_sentiment': sentiment
            })
            
            # Verbose output
            if verbose:
                print(f"\nText [{i+1}/{total}]: {text}")
                print(f"Predicted sentiment: {sentiment}")
                
            # Print progress every 10% if not verbose
            elif (i + 1) % max(1, total // 10) == 0:
                print(f"Progress: {(i + 1) / total:.1%}")
            
            # Save checkpoint at intervals
            if (i + 1) % checkpoint_interval == 0:
                save_checkpoint(i, results)
                
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving checkpoint...")
        save_checkpoint(i, results)
        raise
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Saving checkpoint...")
        save_checkpoint(i, results)
        raise
    
    # Process completed successfully, remove checkpoint file
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Processing completed, checkpoint file removed")
    
    print("Analysis complete!")
    return results

def save_analysis_results(results, model_name, data_name, base_dir="./results"):
    """
    Save analysis results to CSV file with automatic filename generation and collision handling.
    
    Args:
        results: List of dictionaries containing analysis results
        model_name: Name of the model used
        data_name: Name of the dataset used
        base_dir: Directory to save results
    
    Returns:
        str: Path to the saved file
    """
    # Create results directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create base filename
    dataset_prefix = data_name.split('/')[-1][:5]  # Take first 5 letters
    base_filename = f"before_test_{model_name}_{dataset_prefix}"
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Generate filename with collision handling
    filename = f"{base_filename}.csv"
    filepath = os.path.join(base_dir, filename)
    
    if os.path.exists(filepath):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_filename}_{timestamp}.csv"
        filepath = os.path.join(base_dir, filename)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")
    
    return filepath

def load_analysis_results(filepath):
    """
    Load analysis results from CSV file.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        list: List of dictionaries containing analysis results
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    results = df.to_dict('records')
    print(f"Loaded {len(results)} results from: {filepath}")
    
    return results

# Helper function to clean up checkpoints
def cleanup_checkpoints(cache_dir="./cache"):
    """Remove all checkpoint files from the cache directory."""
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            if file.startswith("checkpoint_") and file.endswith(".json"):
                os.remove(os.path.join(cache_dir, file))
        print("Checkpoints cleaned up")

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype="auto",
    cache_dir=cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    add_eos_token=True,
    cache_dir=cache_dir
)

if tokenizer.pad_token_id is None:
    print("No pad token found, setting pad token to eos token")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}

print("\n=== Testing with Instruction Prompts ===")
# results = analyze_sentiment_zero_shot(dataset, model, tokenizer, device)
# results = analyze_sentiment_zero_shot(dataset, model, tokenizer, device, verbose=False, num_samples=20)
try:
    results = analyze_sentiment_zero_shot(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        device=device,
        verbose=False,
        # num_samples=20,
        checkpoint_interval=100,
        model_name=model_name,
        data_name=data_name
    )
except KeyboardInterrupt:
    print("\nProcessing interrupted. You can resume later using the same function call.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("You can resume processing later using the same function call.")


# Save results
results_filepath = save_analysis_results(results, model_name, data_name)