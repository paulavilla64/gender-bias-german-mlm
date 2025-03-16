# Schlachtplan:

# 1. Read in tsv file for Bec-Pro: BEC-Pro_EN.tsv
# 2. Output should be the same tsv-file "results_EN.csv" but with an additional column "Pre_Assoc"
# 3. Include pre-processing steps
# 4. Measure association bias and put it in the column "Pre_Assoc"

import pandas as pd
import math
import random
import time
import numpy as np
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
import torch
from nltk import sent_tokenize
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForMaskedLM
from sklearn.model_selection import train_test_split

from bias_utils.utils import model_evaluation, statistics, input_pipeline, format_time, mask_tokens

print('-- Prepare evaluation data --')

# Read a TSV file
data = pd.read_csv('../BEC-Pro/BEC-Pro_EN.tsv', sep='\t')

# Take only the first 50 rows of data
data = data.head(50)
print("Loaded first 50 rows of the dataset:")
print(data)

# check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# TECHNICAL SPECIFICATIONS AND MODELS
# Use the Huggingface transformers library for PyTorch with a default random seed of 42 for all experiments
# The model used for bias evaluation and fine-tuning is a pre-trained BERTBASE model with a language modelling head on top.
# For English, the tokenizer and model are loaded with the standard pre-trained uncased BERTBASE model.

print('-- Import BERT model --')

# Load the BERT tokenizer and dbmdz model
# model_name_dbmdz = "bert-base-german-dbmdz-cased"
# tokenizer = AutoTokenizer.from_pretrained(model_name_dbmdz)
# model = AutoModelForMaskedLM.from_pretrained(model_name_dbmdz,
#                                             output_attentions=False,
#                                             output_hidden_states=False)

# Load tokenizer and google bert model
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



def fine_tune(model, train_dataloader, val_dataloader, epochs, tokenizer, device):
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
    train_loss_values = []
    val_loss_values = []



    for epoch_i in range(0, epochs):
        print('')
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        model.train()

        # Iterate over batches
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(train_dataloader), elapsed))
            
            print("Raw Input IDs (before masking):", batch[0])

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

            print("Masked Input IDs:", b_input_ids)
            print("Masked Labels:", b_labels)
            print("Attention Mask:", b_input_mask)

            
            print("First 5 values of b_input_ids:", b_input_ids[:5])
            print("First 5 values of b_labels:", b_labels[:5])

            # clear previous gradients
            model.zero_grad()

            # forward pass through the model
            outputs_train = model(b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            print("Model Outputs:", outputs_train)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            # Extract the loss from the model's output tuple
            train_loss = outputs_train[0]
            print(f"Train Loss: {train_loss}")

            # Accumulate the total loss
            total_train_loss += train_loss.item()

            if torch.isnan(train_loss):
                print("❌ Train loss is NaN! Skipping this step.")
                continue

            # Perform a backward pass to calculate the gradients.
            train_loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the 'exploding gradients' problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update model parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data for the epoch
        avg_train_loss = total_train_loss / len(train_dataloader)

        # torch.exp: Returns a new tensor with the exponential of the elements of the input tensor.
        train_perplexity = torch.exp(torch.tensor(avg_train_loss)).item()

        # Store the loss value for plotting the learning curve.
        train_loss_values.append(avg_train_loss)

        print('')
        print(f'\n[Epoch {epoch_i + 1}] Average training loss: {avg_train_loss:.2f}')
        print(f'\n[Epoch {epoch_i + 1}] Training Perplexity: {train_perplexity:.2f}')
        print(f"[Epoch {epoch_i + 1}] Training epoch took: {format_time(time.time() - t0)}")
        
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # # Put the model in evaluation mode--the dropout layers behave differently
        # # during evaluation.
        model.eval()

        # Tracking variables 
        total_val_loss = 0

        # Evaluate data for one epoch
        for batch in val_dataloader:
            
            # Apply masking to input tokens in validation (same as in training)
            b_input_ids, b_labels = mask_tokens(batch[0], tokenizer)

            # extract the variables directly from batch to ensure that new data is used in each validation step instead of
            # reusing value from previous loop from training
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)
            b_input_mask = batch[1].to(device)
            print("Unique labels in validation batch:", torch.unique(b_labels))

            # Check if any tokens were masked
            num_masked = (b_labels != -100).sum().item()
            print(f"Total masked tokens in validation batch: {num_masked}")

            if num_masked == 0:
                print("⚠️ No masked tokens in validation! Loss will be 0. Skipping batch.")
                continue

            # Ensure labels are not all -100 (otherwise loss is meaningless)
            unique_labels = torch.unique(b_labels)
            if torch.all(unique_labels == -100):
                print("⚠️ All labels are -100 in validation! Skipping batch.")
                continue  # Skip this batch
            
            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # NO MASKING during validation
                # The validation set should be unchanged since we evaluate the model on unmodified input.
                outputs_val = model(b_input_ids,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                val_loss = outputs_val[0]
                print(f"Val Loss: {val_loss}")
                total_val_loss += val_loss.item()
        
            if torch.isnan(val_loss):
                print("❌ Val loss is NaN! Skipping this step.")
                continue
            
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        val_loss_values.append(avg_val_loss)

        print(f'\n[Epoch {epoch_i + 1}] Validation Loss: {avg_val_loss:.2f}')
        print(f'\n[Epoch {epoch_i + 1}] Validation Perplexity: {val_perplexity:.2f}')
        print(f'[Epoch {epoch_i + 1}] Validation took: {format_time(time.time() - t0)}')

    print("")

    print('Fine-tuning complete!')

    return model


print('-- Import fine-tuning data --')

# Fine-tune
tune_corpus = pd.read_csv('../Gap/gap_flipped_translated.tsv', sep='\t')
tune_data = []
for text in tune_corpus.Text_German:
    tune_data += sent_tokenize(text)

# make able to handle
# tune_data = tune_data[:50]
# print("Loaded first 50 rows of the finetuning dataset:")
# print(tune_data)

# as max_len get the smallest power of 2 greater or equal to the max sentence length
max_len_tune = max([len(sent.split()) for sent in tune_data])
pos = math.ceil(math.log2(max_len_tune))
max_len_tune = int(math.pow(2, pos))
print('Max len tuning: {}'.format(max_len_tune))

# Split fine-tuning data into train and validation sets
train_texts, val_texts = train_test_split(tune_data, test_size=0.1, random_state=42)

# Tokenize train and validation sets
train_tokens, train_attentions = input_pipeline(tune_data, tokenizer, max_len_tune)
assert train_tokens.shape == train_attentions.shape
val_tokens, val_attentions = input_pipeline(val_texts, tokenizer, max_len_tune)
assert val_tokens.shape == val_attentions.shape


# set up Dataloader
batch_size = 1
train_data = TensorDataset(train_tokens, train_attentions)
# RandomSampler: Used for training to randomly shuffle data for better generalization.
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=batch_size)

val_labels = torch.full_like(val_tokens, -100)  # Ignore index for loss calculation
val_data = TensorDataset(val_tokens, val_attentions, val_labels)
# SequentialSampler: Used for validation to process data in order (no shuffling)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(
    val_data, sampler=val_sampler, batch_size=batch_size)


# for fine-tuning...first masking and then tokenizing
print('-- Set up model fine-tuning --')
epochs = 3
model = fine_tune(model, train_dataloader, val_dataloader, epochs, tokenizer, device)

print('-- Calculate associations after fine-tuning --')
# here tokenization is happening
post_associations = model_evaluation(
    data, tokenizer, model, device)

# add associations to dataframe
data = data.assign(Post_Assoc=post_associations)


# Save the results
output_file = "../data/output_csv_files/german/results_DE_with_perplexity_validation.csv"
data.to_csv(output_file, sep='\t', index=False)
print(f"Results saved to {output_file}")



