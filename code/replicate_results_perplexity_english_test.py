import pandas as pd
import math
import random
import time
import numpy as np
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
import torch
from nltk import sent_tokenize
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForMaskedLM
from bias_utils.utils import model_evaluation, input_pipeline, format_time, mask_tokens

# This training code is based on the run_glue.py script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

print('-- Prepare data --')

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

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

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

model_name_bert = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name_bert)
model = BertForMaskedLM.from_pretrained(model_name_bert,
                                            output_attentions=False,
                                            output_hidden_states=False)

print("loading english bert")
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

def fine_tune(model, train_dataloader, epochs, tokenizer, device):
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

    # For each epoch...
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # train just changes the *mode*, it doesn't *perform* the training.
        # dropout and batchnorm layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # mask inputs so the model can actually learn something
            # Apply masking to input tokens (for masked language modeling)
            # b_labels is generated by mask_tokens(), and we do not use batch[2] at all.
            b_input_ids, b_labels = mask_tokens(batch[0], tokenizer)
            
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # to method.
            #
            # batch contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = b_input_ids.to(device)
            b_input_mask = batch[1].to(device)
            b_labels = b_labels.to(device)  

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the labels.
            # The documentation for this model function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,
                        attention_mask=b_input_mask, 
                        labels=b_labels)
            
            # The call to model always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            train_loss = outputs[0]
            print(f"Train Loss: {train_loss}")

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. loss is a Tensor containing a
            # single value; the .item() function just returns the Python value 
            # from the tensor.
            total_train_loss += train_loss.item()

            # Perform a backward pass to calculate the gradients.
            train_loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_train_loss / len(train_dataloader)   

        # torch.exp: Returns a new tensor with the exponential of the elements of the input tensor.
        train_perplexity = torch.exp(torch.tensor(avg_train_loss)).item()         
        
        # Store the loss value for plotting the learning curve.
        train_loss_values.append(avg_train_loss)

        print("")
        print(f'\n[Epoch {epoch_i + 1}] Average training loss: {avg_train_loss:.2f}')
        print(f'\n[Epoch {epoch_i + 1}] Training Perplexity: {train_perplexity:.2f}')
        print(f"[Epoch {epoch_i + 1}] Training epoch took: {format_time(time.time() - t0)}")
            
        
    print("")
    print("Fine-tuning complete!")

    return model


print('-- Import fine-tuning data --')

# Fine-tune
tune_corpus = pd.read_csv('../data/Gap/gap_flipped.tsv', sep='\t')
tune_data = []
for text in tune_corpus.Text:
    tune_data += sent_tokenize(text)

# make able to handle
tune_data = tune_data[:50]
print("Loaded first 50 rows of the finetuning dataset:")

# as max_len get the smallest power of 2 greater or equal to the max sentence length
max_len_tune = max([len(sent.split()) for sent in tune_data])
pos = math.ceil(math.log2(max_len_tune))
max_len_tune = int(math.pow(2, pos))

eval_corpus = pd.read_csv('../data/ag_news/ag_news_common_crawl.tsv', sep='\t') 
eval_data = []
for text in eval_corpus.Sentence:
    eval_data += sent_tokenize(text)

eval_data = eval_data[:50]
print("Loaded first 50 rows of the validation dataset:")
print(eval_data)

# as max_len get the smallest power of 2 greater or equal to the max sentence length
max_len_eval = max([len(sent.split()) for sent in eval_data])
pos_eval = math.ceil(math.log2(max_len_eval))
max_len_eval = int(math.pow(2, pos_eval))

print(f"Max sentence length in training set: {max_len_tune}")
print(f"Max sentence length in validation set: {max_len_eval}")


# Tokenize train and validation sets
train_tokens, train_attentions = input_pipeline(tune_data, tokenizer, max_len_tune)
assert train_tokens.shape == train_attentions.shape

val_tokens, val_attentions = input_pipeline(eval_data, tokenizer, max_len_eval)
assert val_tokens.shape == val_attentions.shape

# set up Dataloader
batch_size = 1

train_data = TensorDataset(train_tokens, train_attentions)
# RandomSampler: Used for training to randomly shuffle data for better generalization.
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=batch_size)

val_tokens_masked = val_tokens.clone()  # Clone before masking
val_tokens_masked, val_labels = mask_tokens(val_tokens, tokenizer)

# Ensure that padding tokens are ignored in loss computation
val_labels[val_tokens_masked == tokenizer.pad_token_id] = -100  

val_data = TensorDataset(val_tokens_masked, val_attentions, val_labels)

# SequentialSampler: Used for validation to process data in order (no shuffling)
val_sampler = SequentialSampler(val_data)

val_dataloader = DataLoader(
    val_data, sampler=val_sampler, batch_size=50, drop_last=False)


def compute_perplexity(model, val_dataloader, device):
    print("")
    print("Running Validation...")

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
    
    return avg_loss, perplexity


# for fine-tuning...first masking and then tokenizing
print('-- Set up model fine-tuning --')
epochs = 3

# Compute baseline perplexity **before** fine-tuning (without affecting training)
print("\nCalculating baseline perplexity before fine-tuning...")
baseline_loss, baseline_perplexity = compute_perplexity(model, val_dataloader, device)
print(f"Baseline Loss: {baseline_loss:.2f}, Perplexity: {baseline_perplexity:.2f}")

model = fine_tune(model, train_dataloader, epochs, tokenizer, device)

# Compute perplexity separately
eval_losses, perplexities = compute_perplexity(model, val_dataloader, device)

print(f"Perplexity on validation set: {perplexities:.2f}")


# for epoch in range(len(train_losses)):
#     print(f"Epoch {epoch+1}: Train Loss = {train_losses[epoch]:.2f}, Eval Loss = {eval_losses[epoch]:.2f}, Perplexity = {perplexities[epoch]:.2f}")

print('-- Calculate associations after fine-tuning --')
# here tokenization is happening
post_associations = model_evaluation(
    data, tokenizer, model, device)

# add associations to dataframe
data = data.assign(Post_Assoc=post_associations)

# Save the results
output_file = "../data/output_csv_files/english/results_EN_baseline_perplexity_sample_test.csv"
data.to_csv(output_file, sep='\t', index=False)
print(f"Results saved to {output_file}")


