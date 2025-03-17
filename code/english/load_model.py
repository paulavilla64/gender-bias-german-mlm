import pandas as pd
import math
import torch
import numpy as np
import random
import os
from nltk import sent_tokenize
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertForMaskedLM
from bias_utils.utils import model_evaluation, input_pipeline, mask_tokens

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

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')

# Load the model and tokenizer
model_name_bert = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name_bert)
model = BertForMaskedLM.from_pretrained(model_name_bert,
                                       output_attentions=False,
                                       output_hidden_states=False)

# Load the fine-tuned model weights
model_path = "../models/model_without_validation_sample.pt"  # Path to your saved model
model.load_state_dict(torch.load(model_path))
model.to(device)

print("Model loaded successfully!")

# Load the evaluation data
data = pd.read_csv('../BEC-Pro/BEC-Pro_EN.tsv', sep='\t')
# data = data.head(50)  # Same as in your original script
# print("Loaded evaluation data (first 50 rows)")

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
tokenizer = BertTokenizer.from_pretrained(model_name_bert)
baseline_model = BertForMaskedLM.from_pretrained(model_name_bert,
                                       output_attentions=False,
                                       output_hidden_states=False)
baseline_model.to(device)

# Compute baseline perplexity
baseline_loss, baseline_perplexity = compute_perplexity(baseline_model, val_dataloader, device, "Baseline")

# PART 2: Calculate post-fine-tuning metrics
print("\n=== POST-FINE-TUNING METRICS ===")

# Load the fine-tuned model weights
model_path = "../models/model_without_validation_sample.pt"  # Path to your saved model
finetuned_model = BertForMaskedLM.from_pretrained(model_name_bert,
                                       output_attentions=False,
                                       output_hidden_states=False)
finetuned_model.load_state_dict(torch.load(model_path))
finetuned_model.to(device)
print("Fine-tuned model loaded successfully!")

# Calculate post-fine-tuning perplexity
finetuned_loss, finetuned_perplexity = compute_perplexity(finetuned_model, val_dataloader, device, "Fine-tuned")

# Calculate post-association scores
print('Calculating post-association scores...')
post_associations = model_evaluation(data, tokenizer, finetuned_model, device)

# Add post-association scores to the dataframe
data = data.assign(Post_Assoc=post_associations)

# Save the results
output_file = "../data/output_csv_files/english/results_EN_perplexity_workflow.csv"
data.to_csv(output_file, sep='\t', index=False)
print(f"Results saved to {output_file}")

# Print summary statistics of post-association scores
print("\nSummary of post-association scores:")
print(f"Mean: {np.mean(post_associations):.4f}")
print(f"Median: {np.median(post_associations):.4f}")
print(f"Min: {np.min(post_associations):.4f}")
print(f"Max: {np.max(post_associations):.4f}")
print(f"Standard deviation: {np.std(post_associations):.4f}")

# Print perplexity improvement
perplexity_change = ((baseline_perplexity - finetuned_perplexity) / baseline_perplexity) * 100
print(f"\nPerplexity change: {perplexity_change:.2f}% ({'improved' if perplexity_change > 0 else 'worsened'})")