# import pandas as pd
# import random
# import os
# import numpy as np
# import torch
# from transformers import AutoModelForMaskedLM, AutoTokenizer

# from bias_utils.utils import model_evaluation

# print(f"Torch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"CUDA device count: {torch.cuda.device_count()}")
# print(f"Current CUDA device: {torch.cuda.current_device()}")
# print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


# gpu_id = "3"

# # check if GPU is available
# if torch.cuda.is_available():
#     device = torch.device(f"cuda:{gpu_id}")
#     print('We will use the GPU:', torch.cuda.get_device_name(0))
# else:
#     print('No GPU available, using the CPU instead.')
#     device = torch.device('cpu')

# # Set fixed seeds everywhere
# def set_seed(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     # For deterministic operations
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# seed = 116

# set_seed(seed)

# print('-- Prepare evaluation data --')

# # Read a TSV file
# data = pd.read_csv('../BEC-Pro/modified_file_DE_zero_difference.tsv', sep='\t')

# print('-- Load models --')
# # Load your models based on configs
# models_config = [
#     {
#         "name": "dbmdz",
#         "model_id": "bert-base-german-dbmdz-cased",
#         "checkpoint_dir": f"../models/dbmdz_checkpoints/random_seed_{seed}",
#         "checkpoint_file": f"finetuned_dbmdz_{seed}_epoch_3.pt"
#     },
#     {
#         "name": "google_bert",
#         "model_id": "google-bert/bert-base-german-cased",
#         "checkpoint_dir": f"../models/google_bert_checkpoints/random_seed_{seed}",
#         "checkpoint_file": f"finetuned_google_bert_{seed}_epoch_3.pt"
#     },
#     {
#         "name": "deepset_bert",
#         "model_id": "deepset/gbert-base",
#         "checkpoint_dir": f"../models/deepset_bert_checkpoints/random_seed_{seed}",
#         "checkpoint_file": f"finetuned_deepset_bert_{seed}_epoch_3.pt"
#     },
#     {
#         "name": "distilbert",
#         "model_id": "distilbert/distilbert-base-german-cased",
#         "checkpoint_dir": f"../models/distilbert_checkpoints/random_seed_{seed}",
#         "checkpoint_file": f"finetuned_distilbert_{seed}_epoch_3.pt"
#     },
#     {
#         "name": "gelectra",
#         "model_id": "deepset/gelectra-base",
#         "checkpoint_dir": f"../models/gelectra_checkpoints/random_seed_{seed}",
#         "checkpoint_file": f"finetuned_gelectra_{seed}_epoch_3.pt"
#     }
# ]

# # Evaluate each model
# for model_config in models_config:
#     model_name = model_config["name"]
#     model_id = model_config["model_id"]
#     checkpoint_dir = model_config["checkpoint_dir"]
#     checkpoint_file = model_config["checkpoint_file"]

#     print(f"\n{'='*50}")
#     print(f"Processing model: {model_name} ({model_id})")
#     print(f"{'='*50}")

#     try:
#         # Load the base model and tokenizer
#         print(f'-- Loading tokenizer for {model_name} --')
#         tokenizer = AutoTokenizer.from_pretrained(model_id)

#         # Check for checkpoint
#         checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
#         if os.path.exists(checkpoint_path):
#             print(f"Found checkpoint: {checkpoint_path}")

#             # Load the checkpoint
#             print(f'-- Loading {model_name} checkpoint --')
#             model = AutoModelForMaskedLM.from_pretrained(model_id, output_attentions=False, output_hidden_states=False)
#             checkpoint = torch.load(checkpoint_path, map_location=device)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             model.to(device)
#             print("Checkpoint loaded successfully")

#             # Calculate associations with checkpoint model
#             print(f'-- Calculating post associations for {model_name} --')
#             checkpoint_associations = model_evaluation(data, tokenizer, model, device)

#             # Add checkpoint associations to dataframe
#             checkpoint_column = f"{model_name}_Post_Assoc"
#             data[checkpoint_column] = checkpoint_associations
#             print(f"Added Post association scores to column '{checkpoint_column}'")
#         else:
#             print(f"Checkpoint not found: {checkpoint_path}")

#         # Print summary statistics
#         print(f"\nSummary of {model_name} association scores:")
#         print(f"Post-association scores - Mean: {data[checkpoint_column].mean():.4f}, Std: {data[checkpoint_column].std():.4f}")

#     except Exception as e:
#         print(f"Error processing {model_name}: {e}")
#         continue


# # Create output directory if it doesn't exist
# output_dir = "../data/output_csv_files/german"
# os.makedirs(output_dir, exist_ok=True)

# # Save results
# results_file = os.path.join(output_dir, f"post_assoc_all_models_DE_zero_difference_{seed}.csv")
# data.to_csv(results_file, index=False)
# print(f"\nResults saved to {results_file}")

# print("\nFinal summary of all models:")
# for model_config in models_config:
#     model_name = model_config["name"]
#     checkpoint_col = f"{model_name}_Post_Assoc"

#     if checkpoint_col in data.columns:
#         print(f"{model_name} Post association - Mean: {data[checkpoint_col].mean():.4f}, Std: {data[checkpoint_col].std():.4f}")

import pandas as pd
import random
import os
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from bias_utils.utils import model_evaluation

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(
    f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


gpu_id = "4"

# check if GPU is available
if torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_id}")
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')

# Set fixed seeds everywhere


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 47

set_seed(seed)

print('-- Prepare evaluation data --')

# Read a TSV file for model evaluation
data = pd.read_csv('../BEC-Pro/modified_file_DE_gender_neutral.csv', sep='\t')

# Path to the CSV file with pre-association scores
pre_assoc_file = f"../data/output_csv_files/german/pre_assoc_all_models_DE_gender_neutral_{seed}.csv"
print(f'-- Loading pre-association scores from {pre_assoc_file} --')
pre_assoc_data = pd.read_csv(pre_assoc_file)

print('-- Load models --')
# Load your models based on configs
models_config = [
    {
        "name": "dbmdz",
        "model_id": "bert-base-german-dbmdz-cased",
        "checkpoint_dir": f"../models/dbmdz_checkpoints/random_seed_{seed}",
        "checkpoint_file": f"finetuned_dbmdz_{seed}_epoch_3.pt"
    },
    {
        "name": "google_bert",
        "model_id": "google-bert/bert-base-german-cased",
        "checkpoint_dir": f"../models/google_bert_checkpoints/random_seed_{seed}",
        "checkpoint_file": f"finetuned_google_bert_{seed}_epoch_3.pt"
    },
    {
        "name": "deepset_bert",
        "model_id": "deepset/gbert-base",
        "checkpoint_dir": f"../models/deepset_bert_checkpoints/random_seed_{seed}",
        "checkpoint_file": f"finetuned_deepset_bert_{seed}_epoch_3.pt"
    },
    {
        "name": "distilbert",
        "model_id": "distilbert/distilbert-base-german-cased",
        "checkpoint_dir": f"../models/distilbert_checkpoints/random_seed_{seed}",
        "checkpoint_file": f"finetuned_distilbert_{seed}_epoch_3.pt"
    },
    {
        "name": "gelectra",
        "model_id": "deepset/gelectra-base",
        "checkpoint_dir": f"../models/gelectra_checkpoints/random_seed_{seed}",
        "checkpoint_file": f"finetuned_gelectra_{seed}_epoch_3.pt"
    }
]

# Create a dictionary to store post-association scores
post_assoc_scores = {}

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
        print(f'-- Loading tokenizer for {model_name} --')
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Check for checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        if os.path.exists(checkpoint_path):
            print(f"Found checkpoint: {checkpoint_path}")

            # Load the checkpoint
            print(f'-- Loading {model_name} checkpoint --')
            model = AutoModelForMaskedLM.from_pretrained(
                model_id, output_attentions=False, output_hidden_states=False)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            print("Checkpoint loaded successfully")

            # Calculate associations with checkpoint model
            print(f'-- Calculating post associations for {model_name} --')
            checkpoint_associations = model_evaluation(
                data, tokenizer, model, device)

            # Add checkpoint associations to the dictionary
            post_assoc_column = f"{model_name}_Post_Assoc"
            data[post_assoc_column] = checkpoint_associations
            post_assoc_scores[post_assoc_column] = checkpoint_associations
            print(
                f"Added Post association scores to column '{post_assoc_column}'")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")

        # Print summary statistics
        post_assoc_column = f"{model_name}_Post_Assoc"
        if post_assoc_column in data.columns:
            print(f"\nSummary of {model_name} association scores:")
            print(
                f"Post-association scores - Mean: {data[post_assoc_column].mean():.4f}, Std: {data[post_assoc_column].std():.4f}")

    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        continue

# Create output directory if it doesn't exist
output_dir = "../data/output_csv_files/german"
os.makedirs(output_dir, exist_ok=True)

# Save post-association results
post_results_file = os.path.join(
    output_dir, f"post_assoc_all_models_DE_gender_neutral_{seed}.csv")
data.to_csv(post_results_file, index=False)
print(f"\nPost-association results saved to {post_results_file}")

# Now merge the post-association scores with the pre-association data
print("\n-- Merging pre and post association scores --")

# Make sure we have a common identifier column between the two dataframes
# Assuming 'Unnamed: 0' is the common column, but you may need to adjust based on your data
common_column = 'Unnamed: 0'
if common_column not in pre_assoc_data.columns:
    print(
        f"Warning: Column '{common_column}' not found in pre-association file")
    common_columns = set(pre_assoc_data.columns).intersection(
        set(data.columns))
    if common_columns:
        common_column = list(common_columns)[0]
        print(f"Using '{common_column}' as the common column instead")
    else:
        print("Error: No common columns found between the files")
        exit(1)

# Create a merged dataframe starting with the pre-association data
merged_df = pre_assoc_data.copy()

# List to keep track of which columns were added
added_columns = []

# Add each post-association column to the merged dataframe
for model_config in models_config:
    model_name = model_config["name"]
    post_assoc_column = f"{model_name}_Post_Assoc"

    if post_assoc_column in data.columns:
        # Create a mapping from identifier to post-association value
        post_values = dict(zip(data[common_column], data[post_assoc_column]))

        # Add the post-association values to the merged dataframe
        merged_df[post_assoc_column] = merged_df[common_column].map(
            post_values)
        added_columns.append(post_assoc_column)
        print(f"Added column: {post_assoc_column}")

# Save the merged dataframe to a new CSV file
merged_file = os.path.join(
    output_dir, f"post_assoc_all_models_DE_gender_neutral_{seed}.csv")
merged_df.to_csv(merged_file, index=False)
print(f"\nMerged data saved to {merged_file}")

# Print summary of added columns
if added_columns:
    print(
        f"\nSuccessfully added {len(added_columns)} post-association columns to the pre-association data:")
    for col in added_columns:
        print(f"  - {col}")
else:
    print("\nNo post-association columns were added to the pre-association data")

print("\nFinal summary of all models:")
for model_config in models_config:
    model_name = model_config["name"]
    post_assoc_column = f"{model_name}_Post_Assoc"

    if post_assoc_column in merged_df.columns:
        print(
            f"{model_name} post association - Mean: {merged_df[post_assoc_column].mean():.4f}, Std: {merged_df[post_assoc_column].std():.4f}")
