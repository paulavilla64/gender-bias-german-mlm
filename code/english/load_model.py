# Sets up the environment (seeds, device detection)
# Loads the tokenizer
# Prepares the validation data from AG News dataset
# Defines a function to compute perplexity
# Computes baseline perplexity with the original pre-trained BERT
# Loads each checkpoint model from your training directory
# Computes perplexity for each checkpoint
# Saves the results to a CSV file

import pandas as pd
import math
import torch
import numpy as np
import random
import os
import glob
from nltk import sent_tokenize
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertForMaskedLM
from bias_utils.utils import input_pipeline, mask_tokens

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


gpu_id = "4" 

# check if GPU is available
if torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_id}")
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')

# Set all seeds for reproducibility
def set_all_seeds(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(42)


# Load the model and tokenizer
model_name_bert = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name_bert)
print("Tokenizer loaded successfully!")


# Function to compute perplexity
def compute_perplexity(model, val_dataloader, device, description=""):
    print("")
    print(f"\nRunning {description} perplexity calculation...")

    model.eval()  # Set model to evaluation mode
    losses = []
    
    with torch.no_grad():  # No gradients needed
        for batch in val_dataloader:
            inputs, attn_masks, labels = batch
            inputs, attn_masks, labels = inputs.to(device), attn_masks.to(device), labels.to(device)
            
            outputs = model(input_ids=inputs, attention_mask=attn_masks, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)  # Perplexity = exp(loss)

    print(f"{description} Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    return avg_loss, perplexity

# Load and prepare validation data for perplexity calculation
print("Preparing validation data...")
eval_corpus = pd.read_csv('../data/ag_news/ag_news_common_crawl.tsv', sep='\t') 
eval_data = []
for text in eval_corpus.Sentence:
    eval_data += sent_tokenize(text)

# eval_data = eval_data[:50]  # Same as in your original script

# Get max length for padding
max_len_eval = max([len(sent.split()) for sent in eval_data])
pos_eval = math.ceil(math.log2(max_len_eval))
max_len_eval = int(math.pow(2, pos_eval))
print(f"Max sentence length in validation set: {max_len_eval}")

# Tokenize validation set
val_tokens, val_attentions = input_pipeline(eval_data, tokenizer, max_len_eval)
assert val_tokens.shape == val_attentions.shape

# Create masked versions for perplexity
val_tokens_masked = val_tokens.clone()  # Clone before masking
val_tokens_masked, val_labels = mask_tokens(val_tokens, tokenizer)

# Ensure that padding tokens are ignored in loss computation
val_labels[val_tokens_masked == tokenizer.pad_token_id] = -100  

# Create DataLoader
val_data = TensorDataset(val_tokens_masked, val_attentions, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(
    val_data, sampler=val_sampler, batch_size=50, drop_last=False)


# PART 1: Calculate baseline perplexity with pre-trained model
print("\n=== BASELINE METRICS (PRE-TRAINED MODEL) ===")
model_name_bert = "bert-base-uncased"
baseline_model = BertForMaskedLM.from_pretrained(model_name_bert,
                                       output_attentions=False,
                                       output_hidden_states=False)
baseline_model.to(device)

print("Bert model loaded successfully!")

# Compute baseline perplexity
baseline_loss, baseline_perplexity = compute_perplexity(baseline_model, val_dataloader, device, "Baseline")

# Create a simple DataFrame to store results
results = []

# Find all checkpoint files
checkpoints_dir = '../models/bert_checkpoints'
checkpoint_files = sorted(glob.glob(os.path.join(checkpoints_dir, 'finetuned_bert_epoch_*.pt')))

if not checkpoint_files:
    print("No checkpoint files found in directory:", checkpoints_dir)
else:
    print(f"Found {len(checkpoint_files)} checkpoint files")

    # Process each checkpoint
    for checkpoint_file in checkpoint_files:
        checkpoint_name = os.path.basename(checkpoint_file)
        print(f"\n=== Processing checkpoint: {checkpoint_name} ===")
        
        # Load the model
        model = BertForMaskedLM.from_pretrained(model_name_bert,
                                       output_attentions=False,
                                       output_hidden_states=False)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Get epoch number from checkpoint
        epoch = checkpoint['epoch']
        print(f"Checkpoint from epoch {epoch} loaded successfully!")
        
        # Compute perplexity
        val_loss, perplexity = compute_perplexity(model, val_dataloader, device, f"Epoch {epoch}")
        
        # Store results
        results.append({
            'checkpoint': checkpoint_name,
            'epoch': epoch,
            'validation_loss': val_loss,
            'perplexity': perplexity
        })

# Process final model if it exists
final_model_path = '../models/finetuned_bert_final.pt'
if os.path.exists(final_model_path):
    print("\n=== Processing final model ===")
    final_model = BertForMaskedLM.from_pretrained(model_name_bert,
                                        output_attentions=False,
                                        output_hidden_states=False)
    final_model.load_state_dict(torch.load(final_model_path))
    final_model.to(device)

    # Compute final metrics
    final_loss, final_perplexity = compute_perplexity(final_model, val_dataloader, device, "Final Model")
    
    # Store results
    results.append({
        'checkpoint': 'final_model',
        'epoch': 'final',
        'validation_loss': final_loss,
        'perplexity': final_perplexity
    })

print("Final model loaded successfully!")

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_file = "../data/output_csv_files/english/results_padding_EN_with_model_save_epochs_perplexity.csv"
results_df.to_csv(results_file, index=False)
print(f"\nResults saved to {results_file}")

# Print summary
print("\n=== SUMMARY ===")
print(results_df[['checkpoint', 'epoch', 'validation_loss', 'perplexity']])
