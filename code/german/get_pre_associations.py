import torch
import random
import numpy as np
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from bias_utils.utils import model_evaluation


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

seed = 116

set_all_seeds(seed)

print('-- Prepare evaluation data --')

# Read a TSV file
data = pd.read_csv('../BEC-Pro/modified_file_DE_zero_difference.tsv', sep='\t')
print(f"Loaded {len(data)} rows of evaluation data")

# List of models to evaluate
models_config = [
    {
        "name": "dbmdz",
        "model_id": "bert-base-german-dbmdz-cased",
        "checkpoint_dir": f"../models/dbmdz_checkpoints/random_seed_{seed}",
        "checkpoint_file": f"finetuned_dbmdz_{seed}_epoch_0.pt"
    },
    {
        "name": "google_bert",
        "model_id": "google-bert/bert-base-german-cased",
        "checkpoint_dir": f"../models/google_bert_checkpoints/random_seed_{seed}",
        "checkpoint_file": f"finetuned_google_bert_{seed}_epoch_0.pt"
    },
    {
        "name": "deepset_bert",
        "model_id": "deepset/gbert-base",
        "checkpoint_dir": f"../models/deepset_bert_checkpoints/random_seed_{seed}",
        "checkpoint_file": f"finetuned_deepset_bert_{seed}_epoch_0.pt"
    },
    {
        "name": "distilbert",
        "model_id": "distilbert/distilbert-base-german-cased",
        "checkpoint_dir": f"../models/distilbert_checkpoints/random_seed_{seed}",
        "checkpoint_file": f"finetuned_distilbert_{seed}_epoch_0.pt"
    },
    {
        "name": "gelectra",
        "model_id": "deepset/gelectra-base",
        "checkpoint_dir": f"../models/gelectra_checkpoints/random_seed_{seed}",
        "checkpoint_file": f"finetuned_gelectra_{seed}_epoch_0.pt"
    }
]

# Evaluate each model
for model_config in models_config:
    model_name = model_config["name"]
    model_id = model_config["model_id"]
    checkpoint_dir = model_config["checkpoint_dir"]
    checkpoint_file = model_config["checkpoint_file"]
    
    print(f"\n{'='*50}")
    print(f"Processing model: {model_name} ({model_id})")
    print(f"{'='*50}")

    try:
        # Load the base model and tokenizer
        print(f'-- Loading {model_name} model and tokenizer --')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id,
                                                    output_attentions=False,
                                                    output_hidden_states=False)

        # Move model to device
        model.to(device)
        print(f"Model moved to {device}")

        # Calculate pre-association scores with base model
        print(f'-- Calculating pre-association scores for {model_name} --')
        pre_associations = model_evaluation(data, tokenizer, model, device)

        # Add the pre-association scores to dataframe
        column_name = f"{model_name}_Pre_Assoc"
        data[column_name] = pre_associations
        print(f"Added pre-association scores to {len(data)} rows in column '{column_name}'")

        # Check for checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        if os.path.exists(checkpoint_path):
            print(f"Found checkpoint: {checkpoint_path}")
            
            # Load the checkpoint
            print(f'-- Loading {model_name} checkpoint --')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded successfully")

            # Calculate associations with checkpoint model
            print(f'-- Calculating checkpoint associations for {model_name} --')
            checkpoint_associations = model_evaluation(data, tokenizer, model, device)
            
            # Add checkpoint associations to dataframe
            checkpoint_column = f"{model_name}_Checkpoint_Assoc"
            data[checkpoint_column] = checkpoint_associations
            print(f"Added checkpoint association scores to column '{checkpoint_column}'")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            
        # Print summary statistics
        print(f"\nSummary of {model_name} association scores:")
        print(f"Pre-association scores - Mean: {data[column_name].mean():.4f}, Std: {data[column_name].std():.4f}")
        
    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        continue

# Create output directory if it doesn't exist
output_dir = "../data/output_csv_files/german"
os.makedirs(output_dir, exist_ok=True)

# Save results
results_file = os.path.join(output_dir, f"pre_assoc_all_models_DE_zero_difference_{seed}.csv")
data.to_csv(results_file, index=False)
print(f"\nResults saved to {results_file}")

print("\nFinal summary of all models:")
for model_config in models_config:
    model_name = model_config["name"]
    pre_assoc_col = f"{model_name}_Pre_Assoc"
    checkpoint_col = f"{model_name}_Checkpoint_Assoc"
    
    if pre_assoc_col in data.columns:
        print(f"{model_name} pre-association - Mean: {data[pre_assoc_col].mean():.4f}, Std: {data[pre_assoc_col].std():.4f}")
    if checkpoint_col in data.columns:
        print(f"{model_name} checkpoint association - Mean: {data[checkpoint_col].mean():.4f}, Std: {data[checkpoint_col].std():.4f}")