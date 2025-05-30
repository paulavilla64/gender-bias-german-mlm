import torch
import random
import numpy as np
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertTokenizer, BertForMaskedLM
from bias_utils.utils import model_evaluation


import torch
import random
import numpy as np
import os

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Only try to access CUDA devices if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    gpu_id = 4  # Note: should be an integer, not a string
    
    # Check if the requested GPU ID is valid
    if gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
        print(f'Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}')
    else:
        print(f'GPU {gpu_id} not available, using device 0 instead.')
        device = torch.device('cuda:0')
        print(f'Using GPU 0: {torch.cuda.get_device_name(0)}')
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')

# Set all seeds for reproducibility
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_all_seeds(seed)

typ = "neutral"

set_all_seeds(seed)

print('-- Prepare evaluation data --')

# Read a TSV file
data = pd.read_csv(f'../BEC-Pro/BEC-Pro_DE_one_MASK.tsv', sep='\t')
print(f"Loaded {len(data)} rows of evaluation data")

# List of models to evaluate
models_config = [
    {
        "name": "dbmdz",
        "model_id": "bert-base-german-dbmdz-cased",
        "checkpoint_dir": f"../models/Lou/dbmdz_{typ}_checkpoints/random_seed_{seed}",
        "checkpoint_file": f"finetuned_dbmdz_{typ}_{seed}_epoch_0.pt"
    },
    {
        "name": "google-bert",
        "model_id": "google-bert/bert-base-german-cased",
        "checkpoint_dir": f"../models/Lou/google-bert_{typ}_checkpoints/random_seed_{seed}",
        "checkpoint_file": f"finetuned_google-bert_{typ}_{seed}_epoch_0.pt"
    },
    {
        "name": "deepset-bert",
        "model_id": "deepset/gbert-base",
        "checkpoint_dir": f"../models/Lou/deepset-bert_{typ}_checkpoints/random_seed_{seed}",
        "checkpoint_file": f"finetuned_deepset-bert_{typ}_{seed}_epoch_0.pt"
    },
    {
        "name": "distilbert",
        "model_id": "distilbert/distilbert-base-german-cased",
        "checkpoint_dir": f"../models/Lou/distilbert_{typ}_checkpoints/random_seed_{seed}",
        "checkpoint_file": f"finetuned_distilbert_{typ}_{seed}_epoch_0.pt"
    }
    # {   "name": "bert",
    #     "model_id": "bert-base-uncased",
    #     "checkpoint_dir": f"../models/bert_checkpoints/random_seed_{seed}",
    #     "checkpoint_file": f"finetuned_bert_{seed}_epoch_0.pt"
    # }
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
            print(f'-- Calculating pre associations for {model_name} --')
            pre_associations = model_evaluation(data, tokenizer, model, device)
            
            # Add checkpoint associations to dataframe
            pre_column = f"Pre_Assoc_{model_name}"
            data[pre_column] = pre_associations
            print(f"Added pre association scores to column '{pre_column}'")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
        
    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        continue

# Create output directory if it doesn't exist
output_dir = "../data/output_csv_files/german/Lou/pre_assocs/one_mask/"
os.makedirs(output_dir, exist_ok=True)

# Save results
results_file = os.path.join(output_dir, f"pre_assoc_{model_name}_{typ}_DE_regular_one_mask_{seed}.csv")
data.to_csv(results_file, index=False, sep="\t")
print(f"\nResults saved to {results_file}")

print("\nFinal summary of all models:")
for model_config in models_config:
    model_name = model_config["name"]
    pre_col = f"Pre_Assoc_{model_name}"
    
    if pre_col in data.columns:
        print(f"{model_name} checkpoint association - Mean: {data[pre_col].mean():.4f}, Std: {data[pre_col].std():.4f}")