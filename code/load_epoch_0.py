import pandas as pd
import random
import os
import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM
from bias_utils.utils import model_evaluation

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


gpu_id = "3" 

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

set_seed(42) 

print('-- Prepare evaluation data --')

# Read a TSV file
data = pd.read_csv('../BEC-Pro/BEC-Pro_EN.tsv', sep='\t')

# Take only the first 50 rows of data
# data = data.head(50)
# print("Loaded first 50 rows of the dataset:")
# print(data)

# TECHNICAL SPECIFICATIONS AND MODELS
# Use the Huggingface transformers library for PyTorch with a default random seed of 42 for all experiments
# The model used for bias evaluation and fine-tuning is a pre-trained BERTBASE model with a language modelling head on top.
# For English, the tokenizer and model are loaded with the standard pre-trained uncased BERTBASE model.

print('-- Import BERT model --')

model_name_bert = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name_bert)
model = BertForMaskedLM.from_pretrained(model_name_bert,
                                            output_attentions=False,
                                            output_hidden_states=False)

print("loading english bert")

# Verify model and tokenizer
print(f"Tokenizer: {tokenizer}")
print(f"Model loaded: {model_name_bert}")


# PRE-PROCESSING
# As a first step, the fixed input sequence length is determined as the smallest
# power of two greater than or equal to the maximum sequence length. In a second step, the inputs are
# tokenized by the pre-trained BertTokenizer and padded to the previously determined fixed sequence
# length. From the padded and encoded inputs, attention masks are created. Attention mask tensors have
# the same size as the input tensors. For each index of the input tensor, non-pad tokens are marked with a
# 1 and pad tokens with a 0 in the attention mask tensor.

print('-- Calculate associations before fine-tuning --')
# here also tokenization is happening
pre_associations = model_evaluation(data, tokenizer, model, device)

# Add the associations to dataframe
data = data.assign(Pre_Assoc=pre_associations)

# Create directory for model checkpoints
model_dir = '../models/bert_checkpoints/random_seed_42'
os.makedirs(model_dir, exist_ok=True)

# Save the original model as epoch 0 (baseline)
baseline_checkpoint_path = os.path.join(model_dir, 'finetuned_bert_42_epoch_0.pt')
torch.save({
    'epoch': 0,
    'model_state_dict': model.state_dict(),
    # No optimizer or scheduler states for baseline
}, baseline_checkpoint_path)
print(f"Baseline model (epoch 0) saved at {baseline_checkpoint_path}")