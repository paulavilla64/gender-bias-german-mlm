import pandas as pd
import math
import torch
import numpy as np
import random
import os
from nltk import sent_tokenize
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForMaskedLM
from bias_utils.utils import input_pipeline, mask_tokens

# Defines a function to compute perplexity for language models
# Loads validation data from a Leipzig corpus
# Evaluates four German BERT models (dbmdz, google_bert, deepset_bert, and distilbert)
# Compares models across epochs 0-3
# Saves and reports perplexity metrics


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
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
typ = "neutral"

set_all_seeds(seed)

# Function to compute perplexity
def compute_perplexity(model, val_dataloader, device, description=""):
    print("")
    print(f"\nRunning {description} perplexity calculation...")

    model.eval()  
    losses = []
    
    with torch.no_grad():  
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
eval_corpus = pd.read_csv('../data/leipzig_corpus/deu_news_leipzig.tsv', sep='\t') 
eval_data = []
for text in eval_corpus.Sentence:
    eval_data += sent_tokenize(text)

# Get max length for padding
max_len_eval = max([len(sent.split()) for sent in eval_data])
pos_eval = math.ceil(math.log2(max_len_eval))
max_len_eval = int(math.pow(2, pos_eval))
print(f"Max sentence length in validation set: {max_len_eval}")

# List of models to evaluate
models_config = [
    {
        "name": "dbmdz",
        "model_id": "bert-base-german-dbmdz-cased",
        "checkpoint_dir": f"../models/Lou/dbmdz_{typ}_checkpoints/random_seed_{seed}",
    },
    {
        "name": "google-bert",
        "model_id": "google-bert/bert-base-german-cased",
        "checkpoint_dir": f"../models/Lou/google-bert_{typ}_checkpoints/random_seed_{seed}",
    },
    {
        "name": "deepset-bert",
        "model_id": "deepset/gbert-base",
        "checkpoint_dir": f"../models/Lou/deepset-bert_{typ}_checkpoints/random_seed_{seed}",
    },
    {
        "name": "distilbert",
        "model_id": "distilbert/distilbert-base-german-cased",
        "checkpoint_dir": f"../models/Lou/distilbert_{typ}_checkpoints/random_seed_{seed}",
    }
]

# Dictionary to store results for all models
all_results = {}

# Process each model
for model_config in models_config:
    model_name = model_config["name"]
    model_id = model_config["model_id"]
    checkpoint_dir = model_config["checkpoint_dir"]
    
    print(f"\n{'='*50}")
    print(f"Processing model: {model_name} ({model_id})")
    print(f"{'='*50}")
    
    # Load the tokenizer for this model
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Tokenize validation set with the model's tokenizer
    print(f"Tokenizing validation data for {model_name}...")
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
    
    # Create a list to store results for this model
    model_results = []
    
    # Check all epochs from 0 to 3
    for epoch in range(4):  # 0, 1, 2, 3
        checkpoint_file = f"finetuned_{model_name}_{typ}_{seed}_epoch_{epoch}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        
        print(f"\n=== EVALUATING {model_name.upper()} MODEL AT EPOCH {epoch} ===")
        
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            
            # Load the model architecture
            model = AutoModelForMaskedLM.from_pretrained(model_id,
                                         output_attentions=False,
                                         output_hidden_states=False)
            
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Check if the checkpoint contains model_state_dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                # Extract epoch from checkpoint if available
                epoch_from_checkpoint = checkpoint.get('epoch', epoch)
                if epoch_from_checkpoint != epoch:
                    print(f"Warning: Checkpoint contains epoch {epoch_from_checkpoint}, but filename indicates epoch {epoch}")
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            print(f"{model_name} model at epoch {epoch} loaded successfully!")
            
            # Compute perplexity for this epoch
            val_loss, perplexity = compute_perplexity(model, val_dataloader, device, f"{model_name} Epoch {epoch}")
            
            # Store results
            model_results.append({
                'checkpoint': os.path.basename(checkpoint_path),
                'epoch': epoch,
                'validation_loss': val_loss,
                'perplexity': perplexity
            })
            
            # If it's not epoch 0, calculate improvement from baseline
            if epoch > 0 and len(model_results) > 1:
                baseline_perplexity = model_results[0]['perplexity']  # Epoch 0
                improvement = ((baseline_perplexity - perplexity) / baseline_perplexity) * 100
                status = "improved" if improvement > 0 else "worsened"
                print(f"Perplexity {status} by {abs(improvement):.2f}% compared to baseline")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            
            # If it's epoch 0 and checkpoint doesn't exist, use HuggingFace model as fallback
            if epoch == 0:
                print(f"Falling back to HuggingFace model as baseline")
                model = AutoModelForMaskedLM.from_pretrained(model_id,
                                              output_attentions=False,
                                              output_hidden_states=False)
                model.to(device)
                print(f"Base {model_name} model loaded successfully from Hugging Face!")
                
                # Compute baseline perplexity
                val_loss, perplexity = compute_perplexity(model, val_dataloader, device, f"{model_name} Baseline (HuggingFace)")
                
                # Store baseline results
                model_results.append({
                    'checkpoint': 'baseline_huggingface',
                    'epoch': 0,
                    'validation_loss': val_loss,
                    'perplexity': perplexity
                })
    
    # Convert results to DataFrame and save for this model
    if model_results:
        results_df = pd.DataFrame(model_results)
        results_dir = f"../data/output_csv_files/german/Lou/perplexity/{model_name}"
        os.makedirs(results_dir, exist_ok=True)
        results_file = f"{results_dir}/results_DE_{model_name}_{typ}_{seed}_perplexity.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nResults for {model_name} saved to {results_file}")
        
        # Store results for summary
        all_results[model_name] = model_results
    else:
        print(f"No results collected for {model_name}")

# Print final summary for all models
print("\n\n" + "="*70)
print("SUMMARY OF PERPLEXITY RESULTS FOR ALL MODELS")
print("="*70)

for model_name, results in all_results.items():
    if len(results) > 1:  # Only show comparison if we have both baseline and at least one other epoch
        baseline = results[0]['perplexity']  # Epoch 0
        
        print(f"{model_name.upper()}:")
        print(f"  Baseline perplexity (Epoch 0): {baseline:.2f}")
        
        # Report perplexity for all available epochs (1-3)
        for res in results[1:]:
            epoch = res['epoch']
            perplexity = res['perplexity']
            improvement = ((baseline - perplexity) / baseline) * 100
            status = "improved" if improvement > 0 else "worsened"
            
            print(f"  Epoch {epoch} perplexity: {perplexity:.2f} ({status} by {abs(improvement):.2f}%)")
        
        print("-" * 50)
        
        # Find best epoch
        best_result = min(results, key=lambda x: x['perplexity'])
        best_epoch = best_result['epoch']
        best_perplexity = best_result['perplexity']
        
        if best_epoch != 0:  # If best is not baseline
            best_improvement = ((baseline - best_perplexity) / baseline) * 100
            print(f"  Best result: Epoch {best_epoch} with perplexity {best_perplexity:.2f} ({abs(best_improvement):.2f}% improvement)")
        else:
            print(f"  Best result: Baseline (Epoch 0) with perplexity {best_perplexity:.2f}")
        
        print("=" * 50)