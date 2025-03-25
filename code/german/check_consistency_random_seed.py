import torch
import numpy as np
import random
import os
from transformers import AutoModelForMaskedLM
import pandas as pd


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

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the different seeds to check
seeds = [42, 116, 387, 1980]
model_id = "deepset/gelectra-base"

# Store models for each seed
models = {}
checkpoints = {}

# First, let's check the models loaded directly from Hugging Face
print("CHECKING MODELS LOADED DIRECTLY FROM HUGGING FACE")
print("=" * 70)

# Load models with different seeds
for seed in seeds:
    set_all_seeds(seed)
    print(f"Loading model with seed {seed}")
    model = AutoModelForMaskedLM.from_pretrained(model_id, 
                                               output_attentions=False,
                                               output_hidden_states=False)
    models[seed] = model

# Compare models pairwise
results = []
for i, seed1 in enumerate(seeds):
    for seed2 in seeds[i+1:]:
        print(f"\nComparing models with seeds {seed1} vs {seed2}")
        max_diff = 0
        diff_layer = ""
        
        for (name1, param1), (name2, param2) in zip(models[seed1].named_parameters(), 
                                                  models[seed2].named_parameters()):
            # Convert to numpy for easier comparison
            tensor1 = param1.detach().cpu().numpy()
            tensor2 = param2.detach().cpu().numpy()
            
            # Calculate difference
            diff = np.abs(tensor1 - tensor2).max()
            if diff > max_diff:
                max_diff = diff
                diff_layer = name1
                
        print(f"Maximum difference: {max_diff}")
        print(f"Layer with max difference: {diff_layer}")
        results.append({
            "seed1": seed1,
            "seed2": seed2,
            "source": "huggingface",
            "max_diff": max_diff,
            "diff_layer": diff_layer
        })

# Now check checkpoint loading
print("\n\nCHECKING MODELS LOADED FROM CHECKPOINTS")
print("=" * 70)

checkpoint_dir = "../models/gelectra_checkpoints/random_seed_"

# Load checkpoint models with different seeds
for seed in seeds:
    set_all_seeds(seed)
    print(f"Loading checkpoint with seed {seed}")
    
    checkpoint_path = f"{checkpoint_dir}{seed}/finetuned_gelectra_{seed}_epoch_0.pt"
    
    if os.path.exists(checkpoint_path):
        model = AutoModelForMaskedLM.from_pretrained(model_id,
                                                   output_attentions=False,
                                                   output_hidden_states=False)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        checkpoints[seed] = model
        print(f"  Checkpoint loaded: {checkpoint_path}")
    else:
        print(f"  Checkpoint not found: {checkpoint_path}")

# Compare checkpoint models pairwise
if len(checkpoints) > 1:
    checkpoint_seeds = list(checkpoints.keys())
    
    for i, seed1 in enumerate(checkpoint_seeds):
        for seed2 in checkpoint_seeds[i+1:]:
            print(f"\nComparing checkpoint models with seeds {seed1} vs {seed2}")
            max_diff = 0
            diff_layer = ""
            
            for (name1, param1), (name2, param2) in zip(checkpoints[seed1].named_parameters(), 
                                                      checkpoints[seed2].named_parameters()):
                # Convert to numpy for easier comparison
                tensor1 = param1.detach().cpu().numpy()
                tensor2 = param2.detach().cpu().numpy()
                
                # Calculate difference
                diff = np.abs(tensor1 - tensor2).max()
                if diff > max_diff:
                    max_diff = diff
                    diff_layer = name1
                    
            print(f"Maximum difference: {max_diff}")
            print(f"Layer with max difference: {diff_layer}")
            results.append({
                "seed1": seed1,
                "seed2": seed2,
                "source": "checkpoint",
                "max_diff": max_diff,
                "diff_layer": diff_layer
            })

# Print summary
print("\n\nSUMMARY OF DIFFERENCES")
print("=" * 70)
df = pd.DataFrame(results)
print(df)