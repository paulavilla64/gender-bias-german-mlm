"""
German BERT Fine-tuning with Gender-Inclusive Language for Bias Mitigation

This script fine-tunes German BERT models on gender-inclusive language data (Lou dataset) 
and evaluates gender bias before and after training using association-based measurements 
on German BEC-Pro dataset.

Features:
- Loads German BEC-Pro dataset for bias evaluation
- Calculates pre-training gender bias associations using word-level measurements
- Fine-tunes German BERT on Lou gender-inclusive corpus (neutral + gender star texts)
- Processes both neutral and gender star (*) forms for comprehensive training
- Saves model checkpoints after each epoch (including epoch 0 baseline)
- Applies masked language modeling during training with batch monitoring
- Calculates post-training gender bias associations
- Uses fixed random seed for reproducibility

Supported Models:
- DBMDZ BERT (default active)
- Google BERT (commented)
- G-BERT/deepset BERT (commented)
- DistilBERT (commented)

Output:
- Model checkpoints for each training epoch
- CSV file with pre/post association scores for bias analysis
- Final fine-tuned model state

Usage: Uncomment desired model and run to fine-tune German BERT on gender-inclusive data
"""

import pandas as pd
import math
import random
import time
import os
import numpy as np
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
import torch
from nltk import sent_tokenize
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForMaskedLM

from bias_utils.utils import model_evaluation, input_pipeline, format_time, mask_tokens

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


gpu_id = "5" 

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

seed = 42

typ = "star"

set_seed(seed)  

model_name = "dbmdz"

print('-- Prepare evaluation data --')

# Read a TSV file
data = pd.read_csv('../../datasets/BEC-Pro/modified_corpus/BEC-Pro_DE_gender_neutral_adapted.tsv', sep='\t')

# Take only the first 50 rows of data
# data = data.head(50)
# print("Loaded first 50 rows of the dataset:")
# print(data)

# TECHNICAL SPECIFICATIONS AND MODELS
# Use the Huggingface transformers library for PyTorch with a default random seed of 42 for all experiments
# The model used for bias evaluation and fine-tuning is a pre-trained BERTBASE model with a language modelling head on top.
# For English, the tokenizer and model are loaded with the standard pre-trained uncased BERTBASE model.

print(f'-- Import {model_name} model --')

# Load the BERT tokenizer and dbmdz model
model_name_dbmdz = "bert-base-german-dbmdz-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name_dbmdz)
model = AutoModelForMaskedLM.from_pretrained(model_name_dbmdz,
                                            output_attentions=False,
                                            output_hidden_states=False)

#Load tokenizer and google bert model
# model_name_google_bert = "google-bert/bert-base-german-cased"
# tokenizer = AutoTokenizer.from_pretrained(model_name_google_bert)
# model = AutoModelForMaskedLM.from_pretrained(model_name_google_bert, 
#                                              output_attentions=False,
#                                              output_hidden_states=False)
# Load tokenizer and deepset bert model
# model_name_deepset_bert = "deepset/gbert-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name_deepset_bert)
# model = AutoModelForMaskedLM.from_pretrained(model_name_deepset_bert,
#                                              output_attentions=False,
#                                              output_hidden_states=False)

# model_name_distilbert = "distilbert/distilbert-base-german-cased"
# tokenizer = AutoTokenizer.from_pretrained(model_name_distilbert)
# model = AutoModelForMaskedLM.from_pretrained(model_name_distilbert,
#                                              output_attentions=False,
#                                              output_hidden_states=False)

# model_name_gelectra = "deepset/gelectra-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name_gelectra)
# model = AutoModelForMaskedLM.from_pretrained(model_name_gelectra,
#                                              output_attentions=False,
#                                              output_hidden_states=False)

print(f"loading {model_name}")

# Verify model and tokenizer
print(f"Tokenizer: {tokenizer}")
print(f"Model loaded: {model_name}")


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
model_dir = f'../models/Lou/{model_name}_{typ}_checkpoints/random_seed_{seed}'
os.makedirs(model_dir, exist_ok=True)

# Save the original model as epoch 0 (baseline)
baseline_checkpoint_path = os.path.join(model_dir, f'finetuned_{model_name}_{typ}_{seed}_epoch_0.pt')
torch.save({
    'epoch': 0,
    'model_state_dict': model.state_dict(),
    # No optimizer or scheduler states for baseline
}, baseline_checkpoint_path)
print(f"Baseline model (epoch 0) saved at {baseline_checkpoint_path}")

def fine_tune(model, train_dataloader, epochs, tokenizer, device):
    model_dir = f'../models/Lou/{model_name}_{typ}_checkpoints/random_seed_{seed}'

    os.makedirs(model_dir, exist_ok=True)
    
    model.to(device)

    # ##### NEXT part + comments from tutorial:
    # https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP#scrollTo=oCYZa1lQ8Jn8&forceEdit=true
    # &sandboxMode=true
    # Note: AdamW is a class from the huggingface transformers library (as opposed to pytorch) I
    # believe the 'W' stands for 'Weight Decay fix'
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8)  # args.adam_epsilon  - default is 1e-8. Small constant to prevent division by zero in Adam optimizer

    # Total number of training steps is number of batches per epoch * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler that linearly decreases the learning rate
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py, no warmup steps
                                                num_training_steps=total_steps) # Total training steps

    # Store the average loss after each epoch so we can plot them.
    # train_loss_values = []

    for epoch_i in range(0, epochs):
        print('')
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        model.train()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Iterate over batches
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(train_dataloader), elapsed))
            

            # mask inputs so the model can actually learn something
            # Apply masking to input tokens (for masked language modeling)
            b_input_ids, b_labels = mask_tokens(batch[0], tokenizer)

            # Check unique labels (shouldn't all be -100)
            print("Unique labels in batch:", torch.unique(b_labels))

            # Move tensors to the appropriate device (GPU or CPU)
             # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)
            b_input_mask = batch[1].to(device)

            # clear previous gradients
            model.zero_grad()

            # forward pass through the model
            outputs_train = model(b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            # Extract the loss from the model's output tuple
            train_loss = outputs_train[0]

            total_train_loss += train_loss.item()

            # Perform a backward pass to calculate the gradients.
            train_loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the 'exploding gradients' problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update model parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

            # Calculate the average loss over the training data.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # torch.exp: Returns a new tensor with the exponential of the elements of the input tensor.
        # perplexity = torch.exp(torch.tensor(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        # train_loss_values.append(avg_train_loss)

        print('')
        print('  Average training loss: {0:.2f}'.format(avg_train_loss))
        print(f"[Epoch {epoch_i + 1}] Training epoch took: {format_time(time.time() - t0)}")
        
        # Save model after each epoch - save everything needed to resume training
        epoch_checkpoint_path = os.path.join(model_dir, f'finetuned_{model_name}_{typ}_{seed}_epoch_{epoch_i+1}.pt')
        torch.save({
            'epoch': epoch_i+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, epoch_checkpoint_path)
        print(f"Model checkpoint saved at {epoch_checkpoint_path}")
        

    print("")

    print('Fine-tuning complete!')

    return model


print('-- Import fine-tuning data --')

# Fine-tune
tune_corpus = pd.read_csv('../../datasets/Lou/neutral_genderstern_texts.tsv', sep='\t')
# tune_data = []
# for text in tune_corpus.GenderStern_Text:
#     tune_data += sent_tokenize(text)

# Initialize an empty list for your tuning data
tune_data = []

# Assuming you want to process both 'GenderStern_Text' and another column (e.g., 'Original_Text')
for neutral_text, genderstern_text in zip(tune_corpus.Neutral_Text, tune_corpus.GenderStern_Text):
    # Process the first column
    tune_data += sent_tokenize(neutral_text)
    
    # Process the second column
    tune_data += sent_tokenize(genderstern_text)


# make able to handle
# tune_data = tune_data[:50]
# print("Loaded first 50 rows of the finetuning dataset:")
# print(tune_data)

# as max_len get the smallest power of 2 greater or equal to the max sentence length
max_len_tune = max([len(sent.split()) for sent in tune_data])
pos = math.ceil(math.log2(max_len_tune))
max_len_tune = int(math.pow(2, pos))
print('Max len tuning: {}'.format(max_len_tune))


# Tokenize train and validation sets
train_tokens, train_attentions = input_pipeline(tune_data, tokenizer, max_len_tune)
assert train_tokens.shape == train_attentions.shape


# set up Dataloader
batch_size = 1
train_data = TensorDataset(train_tokens, train_attentions)
# RandomSampler: Used for training to randomly shuffle data for better generalization.
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=batch_size)

# for fine-tuning...first masking and then tokenizing
print('-- Set up model fine-tuning --')
epochs = 3
model = fine_tune(model, train_dataloader, epochs, tokenizer, device)

# Save the model state
torch.save(model.state_dict(), f'../models/Lou/finetuned_{model_name}_{typ}_{seed}_final.pt')
print("Final model saved")

print('-- Calculate associations after fine-tuning --')
# here tokenization is happening
post_associations = model_evaluation(
    data, tokenizer, model, device)

# add associations to dataframe
data = data.assign(Post_Assoc=post_associations)

# Save the results
output_file = f"../../results/association_files/german/Lou/{typ}/results_Lou_gender_neutral_{typ}_{model_name}_{seed}.csv"
data.to_csv(output_file, sep='\t', index=False)
print(f"Results saved to {output_file}")