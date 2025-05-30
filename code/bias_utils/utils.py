import datetime
import math
from typing import Tuple
import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import PreTrainedTokenizer


# taken from https://github.com/allenai/dont-stop-pretraining/blob/master/scripts/mlm_study.py
def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability=0.15) -> Tuple[
        torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the "
            "--mlm flag if you want to use this tokenizer. "
        )

    # Create a copy of the input tensor to serve as the labels. This ensures the original input remains unchanged.
    labels = inputs.clone().long()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults
    # to 0.15 in Bert/RoBERTa)

    # Create a tensor of the same shape as labels with all values set to the masking probability (mlm_probability).
    probability_matrix = torch.full(labels.shape, mlm_probability)
    # Identify positions of special tokens (e.g., [CLS], [SEP], [PAD]) in each sequence. These tokens are not masked during MLM.
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]

    # Set the probability of masking special tokens to 0.0 to prevent them from being masked
    probability_matrix.masked_fill_(torch.tensor(
        special_tokens_mask, dtype=torch.bool), value=0.0)

    # If the tokenizer has a padding token, ensure padding tokens are not masked by setting their probability to 0.0
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    # Uses torch.bernoulli to randomly decide which tokens to mask based on the probability_matrix.
    # Tokens not selected for masking have their corresponding positions in labels set to -100.
    # This ensures these positions are ignored during loss calculation.
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # Ensure at least one token is masked per sequence 
    for i in range(masked_indices.shape[0]):  # Iterate over batch dimension
        if masked_indices[i].sum() == 0:  # If no tokens were masked in a sequence
            random_idx = torch.randint(0, masked_indices.shape[1], (1,), device=inputs.device)
            masked_indices[i, random_idx] = True  # Force one token to be masked

    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # print("Total masked tokens per sequence:", masked_indices.sum(dim=1))

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(
        labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random].type_as(inputs)

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def attention_mask_creator(input_ids):
    """Provide the attention mask list of lists: 0 only for [PAD] tokens (index 0)
    Returns torch tensor"""
    attention_masks = []

    # For each sentence in the input_ids
    for sent in input_ids:
        # Create attention mask: 1 for non-padding tokens, 0 for padding tokens
        segments_ids = [int(t > 0) for t in sent]

        attention_masks.append(segments_ids)

    return torch.tensor(attention_masks)


def tokenize_to_id(sentences, tokenizer):
    """Tokenize all of the sentences and map the tokens to their word IDs."""
    input_ids = []

    # Print total number of sentences to be tokenized
    print(f"Number of sentences to tokenize: {len(sentences)}")

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'

            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    print(f"Number of sentences successfully tokenized: {len(input_ids)}")
    return input_ids


def input_pipeline(sequence, tokenizer, MAX_LEN):
    """function to tokenize, pad and create attention masks"""
    # Step 1: Tokenize the input sequence to IDs
    # Print the first 5 sequences for inspection
    # print(f"Input sequence (first 5): {sequence[:5]}")
    input_ids = tokenize_to_id(sequence, tokenizer)
    # Print the first 5 tokenized inputs
    # print(f"Tokenized input IDs (first 5): {input_ids[:5]}")

    # Step 2: Pad the sequences to the specified MAX_LEN
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                              dtype="long",
                              value=tokenizer.pad_token_id,
                              truncating="post", padding="post")
    # Print the first 5 padded sequences
    # print(f"Padded input IDs (first 5): {input_ids[:5]}")

    # Convert to tensor
    input_ids = torch.tensor(input_ids)
    # Print the shape of the tensor
    # print(f"Input IDs as tensor (shape): {input_ids.shape}")

    # Step 3: Create attention masks
    attention_masks = attention_mask_creator(input_ids)
    # Print the first 5 attention masks
    # print(f"Attention masks (first 5): {attention_masks[:5]}")
    # Print the shape of attention masks
    # print(f"Attention masks shape: {attention_masks.shape}")

    return input_ids, attention_masks


# pred_TM: Predictions from BERT when only the target word is masked (target probability)
# pred_TAM: Predictions from BERT when both the target and attribute words are masked (prior probability)
# input_ids_TAM: Input token IDs for the sentence with both the target and attribute masked.
# original_ids: Original token IDs before any masking, used to identify the token ID of target word.
# tokenizer: Tokenizer object to handle special tokens like [MASK].
def prob_with_prior(pred_TM, pred_TAM, input_ids_TAM, original_ids, tokenizer):
    # Probability distribution over all words in BERT's vocabulary for the [MASK] position (target word masked)
    pred_TM = pred_TM.cpu()
    pred_TAM = pred_TAM.cpu()
    input_ids_TAM = input_ids_TAM.cpu()

    probs = []
    for doc_idx, id_list in enumerate(input_ids_TAM):
        print(f"Processing sentence {doc_idx}")
        print(f"Input IDs for sentence {doc_idx}:", input_ids_TAM[doc_idx])
        print(f"Decoded tokens for sentence {doc_idx}:", tokenizer.convert_ids_to_tokens(input_ids_TAM[doc_idx]))

        # see where the masks were placed in this sentence
        # Finds the positions of all [MASK] tokens in the input, e.g., [0, 3, 4]
        mask_indices = np.where(id_list == tokenizer.mask_token_id)[0]
        print(f"Mask indices: {mask_indices}")

        # now get the probability of the target word:
        # first get id of target word
        # Retrieves the token ID for the target word ("He") from the original sentence, using the first [MASK] index (mask_indices[0]), e.g., [0]
        target_id = original_ids[doc_idx][mask_indices[0]]
        print(f"Target word token ID: {target_id}")

        # get its probability with unmasked profession
        # P_t
        target_prob = pred_TM[doc_idx][mask_indices[0]][target_id].item()
        print(f"Target probability (p_T): {target_prob}")

        # get its prior probability (masked profession)
        # p_prior
        prior = pred_TAM[doc_idx][mask_indices[0]][target_id].item()
        print(f"Prior probability (p_prior): {prior} for {target_id}")

        # get the predicted tokens for the masked profession

        # Normalization:
        # Calculate the association by dividing the target probability by the prior and take the natural logarithm
        # By dividing the conditional probability 𝑃_t by the prior P_prior, the approach controls for how likely the target word is in general (independent of the attribute).
        # This normalization is crucial to avoid overestimating associations for very frequent words like "he" or "she."
        print(f"Association score for sentence {doc_idx}: {np.log(target_prob / prior)}")
        probs.append(np.log(target_prob / prior))

        # Logarithmic Transformation: Taking the logarithm helps interpret the results more easily:
        # A positive score indicates a higher association in context than in the prior (target is more likely given the attribute).
        # A negative score indicates a lower association in context than in the prior (target is less likely given the attribute).

    return probs

def model_evaluation(eval_df, tokenizer, model, device):
    """takes professional sentences as DF, a tokenizer & a BERTformaskedLM model
    and predicts the associations"""

    # Step 1: Determine the maximum sequence length.
    # The maximum sequence length is set to the smallest power of 2 greater than or equal to the
    # length of the longest sentence in the Sent_TM column.
    max_len = max([len(sent.split()) for sent in eval_df.Sent_TM])
    pos = math.ceil(math.log2(max_len))
    max_len_eval = int(math.pow(2, pos))
    # This ensures compatibility with model padding and improves computational efficiency.

    print('max_len evaluation: {}'.format(max_len_eval))

    # Step 2: Tokenize the inputs for different scenarios.
    # create BERT-ready inputs: target masked, target and attribute masked,
    # and the tokenized original inputs to recover the original target word
    print('--- Tokenizing Sent_TM...')
    # Tokenize and create attention masks for sentences where the profession word is unmasked.
    eval_tokens_TM, eval_attentions_TM = input_pipeline(eval_df.Sent_TM,
                                                        tokenizer,
                                                        max_len_eval)
    # print(f'Tokens (Sent_TM): {eval_tokens_TM.shape}')
    # print(f'Attention Masks (Sent_TM): {eval_attentions_TM.shape}')

    print('--- Tokenizing Sent_TAM...')
    # Tokenize and create attention masks for sentences where both the target and attribute are masked.
    eval_tokens_TAM, eval_attentions_TAM = input_pipeline(eval_df.Sent_TAM,
                                                          tokenizer,
                                                          max_len_eval)
    # print(f'Tokens (Sent_TAM): {eval_tokens_TAM.shape}')
    # print(f'Attention Masks (Sent_TAM): {eval_attentions_TAM.shape}')

    print('--- Tokenizing Original Sentence...')
    # Tokenize the original sentences to recover the target word for probability calculations later.
    eval_tokens, _ = input_pipeline(eval_df.Sentence, tokenizer, max_len_eval)

    # print(f'Tokens (Original): {eval_tokens.shape}')
    # check that lengths match before going further
    # Step 3: Validate the shapes of the tokenized inputs and attention masks.
    assert eval_tokens_TM.shape == eval_attentions_TM.shape
    assert eval_tokens_TAM.shape == eval_attentions_TAM.shape
    print('Shapes verified for tokenized inputs and attention masks.')

    # Step 4: Create a DataLoader for evaluation.
    eval_batch = 20
    # Combine all tokenized inputs into a TensorDataset for efficient batching.
    eval_data = TensorDataset(eval_tokens_TM, eval_attentions_TM,
                              eval_tokens_TAM, eval_attentions_TAM,
                              eval_tokens)
    # Use a SequentialSampler to iterate through the data in order.
    eval_sampler = SequentialSampler(eval_data)
    # Create a DataLoader for batching and efficient processing.
    eval_dataloader = DataLoader(eval_data, batch_size=eval_batch,
                                 sampler=eval_sampler)

    print('Evaluation DataLoader created.')
    # put everything to GPU (if it is available)
    # eval_tokens_TM = eval_tokens_TM.to(device)
    # eval_attentions_TM = eval_attentions_TM.to(device)
    # eval_tokens_TAM = eval_tokens_TAM.to(device)
    # eval_attentions_TAM = eval_attentions_TAM.to(device)
    # Step 5: Move the model to the specified device (CPU or GPU).
    model.to(device)
    print(f'Model moved to device: {device}')

    # put model in evaluation mode & start predicting
    model.eval()                # Set the model to evaluation mode (disables dropout, etc.)
    # Initialize a list to store association scores for all sentences.
    associations_all = []
    for step, batch in enumerate(eval_dataloader):
        # Move the tokenized inputs and attention masks for the current batch to the device.
        b_input_TM = batch[0].to(device)
        b_att_TM = batch[1].to(device)
        b_input_TAM = batch[2].to(device)
        b_att_TAM = batch[3].to(device)

        with torch.no_grad():   # Disable gradient computation for inference.
            # Forward pass for Sent_TM (target word unmasked).
            outputs_TM = model(b_input_TM,
                               attention_mask=b_att_TM)
            # Forward pass for Sent_TAM (target word and attribute masked).
            outputs_TAM = model(b_input_TAM,
                                attention_mask=b_att_TAM)
            # Apply softmax to convert logits into probability distributions.
            # Shape: [batch_size, seq_len, vocab_size]
            predictions_TM = softmax(outputs_TM[0], dim=2)
            # Shape: [batch_size, seq_len, vocab_size]
            predictions_TAM = softmax(outputs_TAM[0], dim=2)
        
        # Identify [MASK] positions at column TM
        for doc_idx, id_list in enumerate(b_input_TM):
            mask_indices = (id_list == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

            print(f"\nSentence {step * eval_batch + doc_idx}:")
            print(f"Original: {tokenizer.convert_ids_to_tokens(id_list.tolist())}")

            for mask_pos in mask_indices:
                top_predictions = torch.topk(predictions_TAM[doc_idx, mask_pos], k=5)
                predicted_tokens = tokenizer.convert_ids_to_tokens(top_predictions.indices.tolist())

                print(f"For TM: MASK at position {mask_pos}: {predicted_tokens} (top-5 predictions)")
        
        # Identify [MASK] positions at column TAM
        for doc_idx, id_list in enumerate(b_input_TAM):
            mask_indices = (id_list == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

            print(f"\nSentence {step * eval_batch + doc_idx}:")
            print(f"Original: {tokenizer.convert_ids_to_tokens(id_list.tolist())}")

            for mask_pos in mask_indices:
                top_predictions = torch.topk(predictions_TAM[doc_idx, mask_pos], k=5)
                predicted_tokens = tokenizer.convert_ids_to_tokens(top_predictions.indices.tolist())

                print(f"For TAM: MASK at position {mask_pos}: {predicted_tokens} (top-5 predictions)")

        # Verify that the output shapes match between Sent_TM and Sent_TAM.
        assert predictions_TM.shape == predictions_TAM.shape
        print(f'Batch {step}:')

        # Step 7: Calculate association scores for the batch.
        associations = prob_with_prior(predictions_TM,
                                       predictions_TAM,
                                       b_input_TAM,
                                       batch[4],  # normal inputs
                                       tokenizer)
        print(f'Batch {step}: Associations calculated.')

        # Append the batch's association scores to the overall list.
        associations_all += associations

    # Step 8: Return the complete list of association scores.
    print('Evaluation completed.')
    return associations_all


# Helper function for formatting elapsed times.
def format_time(elapsed):
    """ Takes a time in seconds and returns a string hh:mm:ss"""
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))