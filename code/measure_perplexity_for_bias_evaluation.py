import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import traceback

# Setup CUDA
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

gpu_id = "5"  # Adjust as needed

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_id}")
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')

typ = "gender_neutral"

# Load your template dataset
df = pd.read_csv(f'../BEC-Pro/modified_file_DE_gender_neutral_one_MASK.tsv', sep='\t')

print(f"Loaded template dataset with {len(df)} rows")

# Check if Prof_Gender already exists in the dataset
if 'Prof_Gender' in df.columns:
    print("Found existing Prof_Gender column in the dataset. Using provided gender categorization.")
    # Print distribution of profession gender categories
    print("Distribution of profession gender categories:")
    print(df['Prof_Gender'].value_counts())
else:
    # If Prof_Gender doesn't exist, this would be a critical error
    print("ERROR: Prof_Gender column not found in the dataset!")
    print("Make sure your dataset contains the Prof_Gender column with values 'male', 'female', or 'balanced'.")
    # We could exit the script here, but let's continue with a warning
    print("Continuing with caution, but results may be incorrect.")

# Print distribution of profession gender categories
print("Distribution of profession gender categories:")
print(df['Prof_Gender'].value_counts())

name_of_model = "dbmdz"

# Define model configuration and paths
base_model_id = "bert-base-german-dbmdz-cased"  # Base model architecture
#base_model_id = "bert-base-uncased" 
#base_model_id = "google-bert/bert-base-german-cased"
#base_model_id = "deepset/gbert-base"
#base_model_id = "distilbert/distilbert-base-german-cased"



# Define seeds and model paths
seeds = [42, 116, 387, 1980]

# Find all professions with both male and female versions, grouped by profession gender
def find_gender_pairs(df):
    """Find all profession pairs with male and female versions"""
    pairs = []
    
    # Group by English profession name for better matching
    for profession_en, group_data in df.groupby('Profession_EN'):
        male_rows = group_data[group_data['Gender'] == 'male']
        female_rows = group_data[group_data['Gender'] == 'female']
        
        if len(male_rows) > 0 and len(female_rows) > 0:
            # Make sure we have Prof_Gender
            if 'Prof_Gender' not in male_rows.columns:
                print(f"Warning: Prof_Gender column missing for profession {profession_en}")
                continue
                
            pair = {
                'profession': profession_en,
                'male_sentence': male_rows.iloc[0]['Sentence'],
                'female_sentence': female_rows.iloc[0]['Sentence'],
                'male_profession': male_rows.iloc[0]['Profession'],
                'female_profession': female_rows.iloc[0]['Profession'],
                'prof_gender': male_rows.iloc[0]['Prof_Gender']  # Use the Prof_Gender from the dataset
            }
            pairs.append(pair)
    
    return pairs

# Calculate perplexity for a given text
def calculate_perplexity(model, tokenizer, text):
    """Calculate sentence-level perplexity"""
    try:
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            # Using masked language modeling for perplexity calculation
            inputs["labels"] = inputs["input_ids"].clone()
            outputs = model(**inputs)
        
        perplexity = torch.exp(outputs.loss).item()
        return perplexity, token_count
    except Exception as e:
        print(f"Error calculating perplexity for text: {text}")
        print(f"Error: {e}")
        return None, None

# Process profession pairs with a given model
def process_pairs(pairs, model, tokenizer, model_name, seed):
    """Process all profession pairs with the given model and return results"""
    results = []
    
    for pair in tqdm(pairs, desc=f"Processing {model_name} (seed {seed})"):
        profession = pair['profession']
        male_sent = pair['male_sentence']
        female_sent = pair['female_sentence']
        prof_gender = pair['prof_gender']  # Use the profession gender category
        
        male_perplexity, male_tokens = calculate_perplexity(model, tokenizer, male_sent)
        female_perplexity, female_tokens = calculate_perplexity(model, tokenizer, female_sent)
        
        # Skip if perplexity calculation failed
        if male_perplexity is None or female_perplexity is None:
            print(f"Skipping profession {profession} due to perplexity calculation failure")
            continue
        
        bias_score = female_perplexity - male_perplexity
        
        results.append({
            'model': model_name,
            'seed': seed,
            'profession': profession,
            'male_profession': pair['male_profession'],
            'female_profession': pair['female_profession'],
            'male_sentence': male_sent,
            'female_sentence': female_sent,
            'male_perplexity': male_perplexity,
            'female_perplexity': female_perplexity,
            'bias_score': bias_score,
            'male_tokens': male_tokens,
            'female_tokens': female_tokens,
            'token_count_equal': male_tokens == female_tokens,
            'prof_gender': prof_gender  # Include profession gender category
        })
    
    print(f"Processed {len(results)} professions for {model_name} (seed {seed})")
    return pd.DataFrame(results) if results else None

# Find all profession pairs
print("Finding profession pairs...")
profession_pairs = find_gender_pairs(df)
print(f"Found {len(profession_pairs)} profession pairs with both genders")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Process all models across all seeds
all_results = []

for seed in seeds:
    print(f"\n{'='*50}")
    print(f"Processing Seed {seed}")
    print(f"{'='*50}")
    
    try:
        # Load and process baseline model (epoch 0)
        baseline_path = f"../models/Lou/{name_of_model}_both_checkpoints/random_seed_{seed}/finetuned_{name_of_model}_both_{seed}_epoch_0.pt"
        print(f"Loading baseline model from {baseline_path}...")
        
        baseline_model = AutoModelForMaskedLM.from_pretrained(base_model_id)
        checkpoint = torch.load(baseline_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            baseline_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            baseline_model.load_state_dict(checkpoint)
        
        baseline_model.to(device)
        baseline_model.eval()
        
        baseline_results = process_pairs(profession_pairs, baseline_model, tokenizer, "baseline", seed)
        if baseline_results is not None:
            all_results.append(baseline_results)
        
        # Clean up to free memory
        del baseline_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load and process finetuned model
        finetuned_path = f"../models/Lou/final_models/finetuned_{name_of_model}_both_{seed}_final.pt"
        print(f"Loading finetuned model from {finetuned_path}...")
        
        finetuned_model = AutoModelForMaskedLM.from_pretrained(base_model_id)
        checkpoint = torch.load(finetuned_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            finetuned_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            finetuned_model.load_state_dict(checkpoint)
        
        finetuned_model.to(device)
        finetuned_model.eval()
        
        finetuned_results = process_pairs(profession_pairs, finetuned_model, tokenizer, "finetuned", seed)
        if finetuned_results is not None:
            all_results.append(finetuned_results)
        
        # Clean up to free memory
        del finetuned_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"Error processing seed {seed}:")
        traceback.print_exc()
        continue

print("\nCombining results from all models and seeds...")
if not all_results:
    print("No results were collected. Exiting.")
    exit(1)

# Combine all results
combined_results = pd.concat(all_results)
combined_results.to_csv(f'./Lou/perplexity_measure_{typ}/{name_of_model}/all_seeds_bias_{name_of_model}_{typ}_one_MASK_groups.csv', index=False)
print(f"Saved raw results with {len(combined_results)} rows")

# Calculate average results across seeds
print("Calculating averages across seeds...")
avg_results = combined_results.groupby(['model', 'profession', 'prof_gender']).agg({
    'male_perplexity': 'mean',
    'female_perplexity': 'mean',
    'bias_score': ['mean', 'std'],
    'male_profession': 'first',
    'female_profession': 'first',
    'male_sentence': 'first',
    'female_sentence': 'first'
}).reset_index()

# Flatten the multi-level columns
avg_results.columns = ['_'.join(col).strip('_') for col in avg_results.columns.values]

# Rename for clarity
avg_results = avg_results.rename(columns={
    'bias_score_mean': 'avg_bias_score',
    'bias_score_std': 'std_bias_score'
})

# Calculate aggregate perplexity statistics by profession gender group
print("Calculating perplexity statistics by profession gender group...")
prof_gender_stats = combined_results.groupby(['model', 'prof_gender']).agg({
    'male_perplexity': ['mean', 'std', 'count'],
    'female_perplexity': ['mean', 'std', 'count'],
    'bias_score': ['mean', 'std']
}).reset_index()

# Flatten the multi-level columns
prof_gender_stats.columns = ['_'.join(col).strip('_') for col in prof_gender_stats.columns.values]

# Save averaged results
avg_results.to_csv(f'./Lou/perplexity_measure_{typ}/{name_of_model}/averaged_bias_{name_of_model}_{typ}_one_MASK_groups.csv', index=False)
print("Saved averaged results")

print("Analysis complete! Results saved to CSV files.")