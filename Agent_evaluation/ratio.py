from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import logging
import csv
import os
from scipy import stats
import ot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate(text, tokenizer, model, device):
    """Calculate political bias distribution using pre-trained model"""
    try:
        # Handle empty text
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for bias calculation")
            return {"left": 0.33, "center": 0.33, "right": 0.33}
            
        # Tokenize with truncation to handle longer sequences
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        # Get confidence scores using softmax
        confidence_scores = logits.softmax(dim=-1)[0].cpu().tolist()

        # Map scores to classes
        classes = ["left", "center", "right"]
        results = dict(zip(classes, confidence_scores))
        
        # Print scores with labels
        for cls, score in results.items():
            logger.info(f"{cls}: {score:.4f}")
            
        return results
        
    except Exception as e:
        logger.error(f"Error in bias calculation: {e}")
        # Return balanced distribution on error
        return {"left": 0.33, "center": 0.33, "right": 0.33}

def normalize_dict(d):
    """Ensure dictionary values sum to 1.0"""
    total = sum(d.values())
    return {k: v / total for k, v in d.items()} if total != 1.0 else d

def align_dicts(dict1, dict2):
    """Ensure both dictionaries have the same keys (case-insensitive)"""
    # Create mapping of lowercase -> original case for both dicts
    keys1_lower = {k.lower(): k for k in dict1.keys()}
    keys2_lower = {k.lower(): k for k in dict2.keys()}
    
    # Get the union of all keys (lowercase)
    all_keys_lower = set(keys1_lower.keys()) | set(keys2_lower.keys())
    
    # Create new dicts with aligned keys
    aligned1 = {}
    aligned2 = {}
    
    for k_lower in all_keys_lower:
        # Use original key if available, otherwise use lowercase
        k1 = keys1_lower.get(k_lower, k_lower)
        k2 = keys2_lower.get(k_lower, k_lower)
        
        # Add to aligned dicts (with 0 if key doesn't exist in original)
        aligned1[k_lower] = dict1.get(k1, 0)
        aligned2[k_lower] = dict2.get(k2, 0)
    
    return normalize_dict(aligned1), normalize_dict(aligned2)

def kl_divergence(dict1, dict2):
    """Calculate KL divergence: KL(dict1 || dict2)"""
    aligned1, aligned2 = align_dicts(dict1, dict2)
    
    # Convert to arrays in the same order
    keys = sorted(aligned1.keys())
    p = np.array([aligned1[k] for k in keys])
    q = np.array([aligned2[k] for k in keys])
    
    # Avoid division by zero in log
    epsilon = 1e-10
    q = np.maximum(q, epsilon)
    
    return np.sum(p * np.log(p / q + epsilon))

def wasserstein_distance(dict1, dict2):
    """Calculate Wasserstein distance between two distributions"""
    aligned1, aligned2 = align_dicts(dict1, dict2)
    
    # Convert to arrays in the same order
    keys = sorted(aligned1.keys())
    p = np.array([aligned1[k] for k in keys])
    q = np.array([aligned2[k] for k in keys])
    
    # Use optimal transport with uniform costs between bins
    # because the other approach would treat the categories as if they were ordered on a line
    # we treat them as unordered
    n = len(keys)
    cost_matrix = np.ones((n, n)) - np.eye(n)  # Cost 1 to move between different bins, 0 for same bin
    return ot.emd2(p, q, cost_matrix)

import html

def load_data(tweets_path, summary):
    tweets = []
    with open(tweets_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            tweets.append(line)
    return tweets, summary

def compute_input_dist(tweets, tokenizer, model, device):
    totals = {"left": 0.0, "center": 0.0, "right": 0.0}
    for tweet in tweets:
        dist = calculate(tweet, tokenizer, model, device)
        for k in totals:
            totals[k] += dist[k]
    n = len(tweets)
    return {k: v / n for k, v in totals.items()}

def load_model(model_name="bucketresearch/politicalBiasBERT", tokenizer_name="bert-base-cased", use_cache=True):
    """Load model and tokenizer with caching option"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    cache_dir = os.path.join(os.getcwd(), "model_cache")
    
    # Create cache directory if it doesn't exist
    if use_cache and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Load tokenizer and model
    if use_cache:
        logger.info(f"Loading tokenizer and model with cache at {cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        logger.info("Loading tokenizer and model without cache")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    return tokenizer, model.to(device), device

def rat(input_path, summary, subject, result_path):
    tokenizer, model, device = load_model(use_cache=True)
    tweets, summary_text = load_data(input_path, summary)

    input_dist = compute_input_dist(tweets, tokenizer, model, device)
    output_dist = calculate(summary_text, tokenizer, model, device)

    kl = kl_divergence(input_dist, output_dist)
    wd = wasserstein_distance(input_dist, output_dist)

    rows = []
    for k in ['left', 'center', 'right']:
        rows.append({'metric': 'input_distribution', 'group': k,
                     'value': round(float(input_dist[k]), 4), 'significance': ''})
    for k in ['left', 'center', 'right']:
        rows.append({'metric': 'output_distribution', 'group': k,
                     'value': round(float(output_dist[k]), 4), 'significance': ''})
    rows.append({'metric': 'kl_divergence', 'group': '',
                 'value': round(float(kl), 4), 'significance': ''})
    rows.append({'metric': 'wasserstein_distance', 'group': '',
                 'value': round(float(wd), 4), 'significance': ''})

    out_path = f'{result_path}.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'group', 'value', 'significance'])
        writer.writeheader()
        writer.writerows(rows)