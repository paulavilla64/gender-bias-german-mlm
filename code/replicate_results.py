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
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
import torch
from nltk import sent_tokenize
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup

from bias_utils.utils import model_evaluation, statistics, input_pipeline, format_time, mask_tokens

print('-- Prepare evaluation data --')

# Read a TSV file
data = pd.read_csv('../BEC-Pro/BEC-Pro_EN.tsv', sep='\t')

# Take only the first 50 rows of data
# data = data.head(10)
# print("Loaded first 10 rows of the dataset:")
# print(data)

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

# Load the BERT tokenizer and model (uncased BERTBASE)
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

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
pre_associations = model_evaluation(data, tokenizer, model, device)

# Add the associations to dataframe
data = data.assign(Pre_Assoc=pre_associations)


def fine_tune(model, dataloader, epochs, tokenizer, device):
    model.to(device)
    model.train()

    # ##### NEXT part + comments from tutorial:
    # https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP#scrollTo=oCYZa1lQ8Jn8&forceEdit=true
    # &sandboxMode=true
    # Note: AdamW is a class from the huggingface transformers library (as opposed to pytorch) I
    # believe the 'W' stands for 'Weight Decay fix'
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8)  # args.adam_epsilon  - default is 1e-8.

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    for epoch_i in range(0, epochs):
        print('')
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        for step, batch in enumerate(dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(dataloader), elapsed))

            # mask inputs so the model can actually learn something
            b_input_ids, b_labels = mask_tokens(batch[0], tokenizer)
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)
            b_input_mask = batch[1].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]

            total_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the 'exploding gradients' problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # torch.exp: Returns a new tensor with the exponential of the elements of the input tensor.
        # perplexity = torch.exp(torch.tensor(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print('')
        print('  Average training loss: {0:.2f}'.format(avg_train_loss))
        print('  Training epoch took: {:}'.format(
            format_time(time.time() - t0)))

    print('Fine-tuning complete!')

    return model


print('-- Import fine-tuning data --')

# Fine-tune
tune_corpus = pd.read_csv('../data/gap_flipped.tsv', sep='\t')
tune_data = []
for text in tune_corpus.Text:
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

# get tokens and attentions tensor for fine-tuning data
tune_tokens, tune_attentions = input_pipeline(
    tune_data, tokenizer, max_len_tune)
assert tune_tokens.shape == tune_attentions.shape

# set up Dataloader
batch_size = 1
train_data = TensorDataset(tune_tokens, tune_attentions)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=batch_size)


print('-- Set up model fine-tuning --')
epochs = 3
model = fine_tune(model, train_dataloader, epochs, tokenizer, device)

print('-- Calculate associations after fine-tuning --')
# calculate associations after fine-tuning
post_associations = model_evaluation(
    data, tokenizer, model, device)

# add associations to dataframe
data = data.assign(Post_Assoc=post_associations)

# if 'Prof_Gender' in data.columns:
#     # divide by gender of person term
#     eval_m = data.loc[data.Prof_Gender == 'male']
#     eval_f = data.loc[data.Prof_Gender == 'female']

print('-- Statistics--')
statistics(data.Pre_Assoc, data.Post_Assoc, data)

# print('-- Statistics After --')
# statistics(eval_f.Post_Assoc, eval_m.Post_Assoc)

# Save the results
output_file = "../data/epochs/replicated_results_epoch3.csv"
data.to_csv(output_file, sep='\t', index=False)
print(f"Results saved to {output_file}")
