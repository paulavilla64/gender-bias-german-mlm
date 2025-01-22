# File name: BERT_utils.py
# Description: additional functionality for my BERT scripts to keep them (relatively) small
# Author: Marion Bartl
# Date: 03/03/2020
import datetime
import math
from typing import Tuple

import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from scipy import stats
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import PreTrainedTokenizer


# Just needed for fine-tuning...check this function
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
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

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


# This function makes sense
def attention_mask_creator(input_ids):
    """Provide the attention mask list of lists: 0 only for [PAD] tokens (index 0)
    Returns torch tensor"""
    attention_masks = []

    # Print the input IDs for the first few sentences to check what's being passed
    # print(f"Number of input sentences: {len(input_ids)}")
    # Print the first sentence's input IDs
    # print(f"First sentence (input IDs): {input_ids[0]}")

    # For each sentence in the input_ids
    for sent in input_ids:
        # Create attention mask: 1 for non-padding tokens, 0 for padding tokens
        segments_ids = [int(t > 0) for t in sent]

        attention_masks.append(segments_ids)

    # Final check: Print the shape of the attention masks tensor
    # print(f"Final attention masks shape: {torch.tensor(attention_masks).shape}")

    return torch.tensor(attention_masks)


def statistics(group1, group2, data):
    """take 2 groups of paired samples and compute either a paired samples t-test or
    a Wilcoxon signed rank test
    prints out a description of the two groups as well as the statistic and p value of the test

    English:
    Group1: Pre-association scores

    Group2: Post-association scores

    German:
    Group1: Pre-association scores for female person words with statistically female professions 
    and male person words with statistically female professions (eval_f)

    Group2: Pre-association scores for male person words with statistically male professions 
    and female person words with statistically male professions (eval_m)"""
    assert len(group1) == len(
        group2), "The two groups do not have the same length"

    print('Group 1:')
    print(group1.describe())
    print('Group 2:')
    print(group2.describe())

    female_professions_pre = data.loc[(
        data['Prof_Gender'] == 'female'), 'Pre_Assoc']
    male_professions_pre = data.loc[(
        data['Prof_Gender'] == 'male'), 'Pre_Assoc']
    balanced_professions_pre = data.loc[(
        data['Prof_Gender'] == 'balanced'), 'Pre_Assoc']

    female_professions_post = data.loc[(
        data['Prof_Gender'] == 'female'), 'Post_Assoc']
    male_professions_post = data.loc[(
        data['Prof_Gender'] == 'male'), 'Post_Assoc']
    balanced_professions_post = data.loc[(
        data['Prof_Gender'] == 'balanced'), 'Post_Assoc']

    print("Female Professions Pre-Test:\n", female_professions_pre.head())
    print("Female Professions Post-Test:\n", female_professions_post.head())
    print("Male Professions Pre-Test:\n", male_professions_pre.head())
    print("Male Professions Post-Test:\n", male_professions_post.head())
    print("Balanced Professions Pre-Test:\n", balanced_professions_pre.head())
    print("Balanced Professions Post-Test:\n",
          balanced_professions_post.head())

    # print("Female Data Equals Male Data:",
    #       female_professions_pre.equals(male_professions_pre))
    # print("Female Data Equals Balanced Data:",
    #       female_professions_pre.equals(balanced_professions_pre))
    # print("Male Data Equals Balanced Data:",
    #       male_professions_pre.equals(balanced_professions_pre))

    print("Statistics before fine-tuning:")
    # Before Fine-tuning
    # Compute mean of the female gender within statistically female professions
    female_gender_in_female_professions_mean = data.loc[(
        data['Prof_Gender'] == 'female') & (data['Gender'] == 'female'), 'Pre_Assoc'].mean()
    print(f'Mean for female gender within statistically female professions: {female_gender_in_female_professions_mean}')

    # Compute mean of the male gender within statistically female professions
    male_gender_in_female_professions_mean = data.loc[(
        data['Prof_Gender'] == 'female') & (data['Gender'] == 'male'), 'Pre_Assoc'].mean()
    print(f'Mean for male gender within statistically female professions: {male_gender_in_female_professions_mean}')

    # Compute mean of female gender within statistically male professions
    female_gender_in_male_professions_mean = data.loc[
        (data['Prof_Gender'] == 'male') & (data['Gender'] == 'female'),
        'Pre_Assoc'
    ].mean()
    print(f'Mean for female gender within statistically male professions: {female_gender_in_male_professions_mean}')

    # Compute mean of male gender within statistically male professions
    male_gender_in_male_professions_mean = data.loc[
        (data['Prof_Gender'] == 'male') & (data['Gender'] == 'male'),
        'Pre_Assoc'
    ].mean()
    print(f'Mean for male gender within statistically male professions: {male_gender_in_male_professions_mean}')

    # Compute mean of female gender within statistically balanced professions
    female_gender_in_balanced_professions_mean = data.loc[
        (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'female'),
        'Pre_Assoc'
    ].mean()
    print(f'Mean for female gender within statistically balanced professions: {female_gender_in_balanced_professions_mean}')

    # Compute mean of male gender within statistically balanced professions
    male_gender_in_balanced_professions_mean = data.loc[
        (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'male'),
        'Pre_Assoc'
    ].mean()
    print(f'Mean for male gender within statistically balanced professions: {male_gender_in_balanced_professions_mean}')

    # After Fine-tuning
    print("Statistics after fine-tuning:")

    # Compute mean of the female gender within statistically female professions
    female_gender_in_female_professions_mean_post = data.loc[
        (data['Prof_Gender'] == 'female') & (data['Gender'] == 'female'),
        'Post_Assoc'].mean()
    print(f'Mean for female gender within statistically female professions: {female_gender_in_female_professions_mean_post}')

    # Compute mean of the male gender within statistically female professions
    male_gender_in_female_professions_mean_post = data.loc[
        (data['Prof_Gender'] == 'female') & (data['Gender'] == 'male'),
        'Post_Assoc'].mean()
    print(f'Mean for male gender within statistically female professions: {male_gender_in_female_professions_mean_post}')

    # Compute mean of female gender within statistically male professions
    female_gender_in_male_professions_mean_post = data.loc[
        (data['Prof_Gender'] == 'male') & (data['Gender'] == 'female'),
        'Post_Assoc'
    ].mean()
    print(f'Mean for female gender within statistically male professions: {female_gender_in_male_professions_mean_post}')

    # Compute mean of male gender within statistically male professions
    male_gender_in_male_professions_mean_post = data.loc[
        (data['Prof_Gender'] == 'male') & (data['Gender'] == 'male'),
        'Post_Assoc'
    ].mean()
    print(f'Mean for male gender within statistically male professions: {male_gender_in_male_professions_mean_post}')

    # Compute mean of female gender within statistically balanced professions
    female_gender_in_balanced_professions_mean_post = data.loc[
        (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'female'),
        'Post_Assoc'
    ].mean()
    print(f'Mean for female gender within statistically balanced professions: {female_gender_in_balanced_professions_mean_post}')

    # Compute mean of male gender within statistically balanced professions
    male_gender_in_balanced_professions_mean_post = data.loc[
        (data['Prof_Gender'] == 'balanced') & (data['Gender'] == 'male'),
        'Post_Assoc'
    ].mean()
    print(f'Mean for male gender within statistically balanced professions: {male_gender_in_balanced_professions_mean_post}')

    # Compute difference score between pre- and post association scores
    diff_F_f = female_gender_in_female_professions_mean_post - \
        female_gender_in_female_professions_mean
    print(f"Difference score for F and f: {diff_F_f}")

    diff_F_m = male_gender_in_female_professions_mean_post - \
        male_gender_in_female_professions_mean
    print(f"Difference score for F and m: {diff_F_m}")

    diff_M_f = female_gender_in_male_professions_mean_post - \
        female_gender_in_male_professions_mean
    print(f"Difference score for M and f: {diff_M_f}")

    diff_M_m = male_gender_in_male_professions_mean_post - \
        male_gender_in_male_professions_mean
    print(f"Difference score for M and m: {diff_M_m}")

    diff_B_f = female_gender_in_balanced_professions_mean_post - \
        female_gender_in_balanced_professions_mean
    print(f"Difference score for B and f: {diff_B_f}")

    diff_B_m = male_gender_in_balanced_professions_mean_post - \
        male_gender_in_balanced_professions_mean
    print(f"Difference score for B and m: {diff_B_m}")

    # Compute average difference score

    # Compute mean of statistically female professions
    female_professions_mean = data.loc[data['Prof_Gender']
                                       == 'female', 'Pre_Assoc'].mean()
    print(
        f'Mean for statistically female professions - PRE: {female_professions_mean}')

    # Compute mean for statistically male professions
    male_professions_mean = data.loc[data['Prof_Gender']
                                     == 'male', 'Pre_Assoc'].mean()
    print(
        f'Mean for statistically male professions - PRE: {male_professions_mean}')

    # Compute mean of statistically balanced professions
    balanced_professions_mean = data.loc[data['Prof_Gender']
                                         == 'balanced', 'Pre_Assoc'].mean()
    print(
        f'Mean for statistically balanced professions - PRE: {balanced_professions_mean}')

    # Compute mean of statistically female professions - POST
    female_professions_mean_post = data.loc[data['Prof_Gender']
                                            == 'female', 'Post_Assoc'].mean()
    print(
        f'Mean for statistically female professions - POST: {female_professions_mean_post}')

    # Compute mean for statistically male professions - POST
    male_professions_mean_post = data.loc[data['Prof_Gender']
                                          == 'male', 'Post_Assoc'].mean()
    print(
        f'Mean for statistically male professions - POST: {male_professions_mean_post}')

    # Compute mean for statistically balanced professions - POST
    balanced_professions_mean_post = data.loc[data['Prof_Gender']
                                              == 'balanced', 'Post_Assoc'].mean()
    print(
        f'Mean for statistically balanced professions - POST: {balanced_professions_mean_post}')

    # Compute difference score for F
    dif_F = female_professions_mean_post - female_professions_mean
    print(f"Difference score for F: {dif_F}")

    # Compute difference scores based on individual values from the Pre_Assoc and Post_Assoc columns
    dif_F_ind = female_professions_pre.sub(
        female_professions_post, fill_value=0)
    print(f"Individual difference score for F_ind: {dif_F_ind}")
    # print("Ranks for F differences:", stats.rankdata(np.abs(dif_F_ind)))

    # Compute Wilcoxon Test for F
    SW_stat_F, SW_p_F = stats.shapiro(dif_F_ind)

    if SW_p_F >= 0.05:
        print('T-Test for F:')
        statistic_F, p_F = stats.ttest_rel(
            female_professions_pre, female_professions_post)
    else:
        print('Wilcoxon Test for F:')
        statistic_F, p_F = stats.wilcoxon(
            female_professions_pre, female_professions_post)

    print('W_F: {}, p: {}'.format(statistic_F, p_F))

    # effect_size_F = statistic_F / np.sqrt(len(female_professions_pre))
    # print('Effect size r_F: {}'.format(effect_size_F))

    # Compute effect size for F (revised)
    n_F = len(female_professions_pre)
    print('Size of dataset n_F: {}'.format(n_F))
    W_expected_F = n_F * (n_F + 1) / 4
    print('Expected W_F: {}'.format(W_expected_F))

    # Compute the standard deviation of W
    SE_W = math.sqrt((n_F * (n_F + 1) * (2 * n_F + 1)) / 24)

    # effect_size_F = (statistic_F - W_expected_F) / \
    #     np.sqrt(n_F * (n_F + 1) * (2 * n_F + 1) / 6)
    # print('Adjusted Effect size r_F: {}'.format(effect_size_F))

    # Compute effect size z-based
    # Compute z-statistic
    z_F = (statistic_F - W_expected_F) / SE_W

    # Compute effect size
    effect_size_F_z_based = z_F / math.sqrt(n_F)
    print(f"Effect size r_F - z-based: {effect_size_F_z_based}")

    # Compute difference score for M
    dif_M = male_professions_mean_post - male_professions_mean
    print(f"Difference score for M: {dif_M}")

    # Compute difference scores based on individual values from the Pre_Assoc and Post_Assoc columns
    dif_M_ind = male_professions_pre.sub(male_professions_post, fill_value=0)
    print(f"Individual difference score for M_ind: {dif_M_ind}")
    # print("Ranks for M differences:", stats.rankdata(np.abs(dif_M_ind)))

    # Compute Wilcoxon Test for M
    SW_stat_M, SW_p_M = stats.shapiro(dif_M_ind)

    if SW_p_M >= 0.05:
        print('T-Test for M:')
        statistic_M, p_M = stats.ttest_rel(
            male_professions_pre, male_professions_post)
    else:
        print('Wilcoxon Test for M:')
        statistic_M, p_M = stats.wilcoxon(
            male_professions_pre, male_professions_post)

    print('W_M: {}, p: {}'.format(statistic_M, p_M))

    # effect_size_M = statistic_M / np.sqrt(len(male_professions_pre))
    # print('Effect size r_M: {}'.format(effect_size_M))

    # Compute effect size for M (revised)
    n_M = len(male_professions_pre)
    print('Size of dataset n_M: {}'.format(n_M))
    W_expected_M = n_M * (n_M + 1) / 4
    print('Expected W_M: {}'.format(W_expected_M))
    # effect_size_M = (statistic_M - W_expected_M) / SE_W
    # print('Adjusted Effect size r_M: {}'.format(effect_size_M))

    # Compute effect size z-based
    # Compute z-statistic
    z_M = (statistic_M - W_expected_M) / SE_W

    # Compute effect size
    effect_size_M_z_based = z_M / math.sqrt(n_M)
    print(f"Effect size r_M - z-based: {effect_size_M_z_based}")

    # Check descriptive statistics
    print("Descriptive statistics M_pre:")
    print(male_professions_pre.describe())
    print("Descriptive statistics M_post:")
    print(male_professions_post.describe())

    # Compute difference score for B
    dif_B = balanced_professions_mean_post - balanced_professions_mean
    print(f"Difference score for B: {dif_B}")

    # Compute difference scores based on individual values from the Pre_Assoc and Post_Assoc columns
    dif_B_ind = balanced_professions_pre.sub(
        balanced_professions_post, fill_value=0)
    print(f"Individual difference score for B_ind: {dif_B_ind}")
    # print("Ranks for B differences:", stats.rankdata(np.abs(dif_B_ind)))

    # Compute Wilcoxon Test for B
    SW_stat_B, SW_p_B = stats.shapiro(dif_B_ind)

    if SW_p_B >= 0.05:
        print('T-Test for B:')
        statistic_B, p_B = stats.ttest_rel(
            balanced_professions_pre, balanced_professions_post)
    else:
        print('Wilcoxon Test for B:')
        statistic_B, p_B = stats.wilcoxon(
            balanced_professions_pre, balanced_professions_post)

    print('W_B: {}, p: {}'.format(statistic_B, p_B))

    # effect_size_B = statistic_B / np.sqrt(len(balanced_professions_pre))
    # print('Effect size r_B: {}'.format(effect_size_B))

    # Compute effect size for B (revised)
    n_B = len(balanced_professions_pre)
    print('Size of dataset n_B: {}'.format(n_B))
    W_expected_B = n_B * (n_B + 1) / 4

    print('Expected W_B: {}'.format(W_expected_B))
    # effect_size_B = (statistic_B - W_expected_B) / SE_W
    # print('Adjusted Effect size r_B: {}'.format(effect_size_B))

    # Compute effect size z-based
    # Compute z-statistic
    z_B = (statistic_B - W_expected_B) / SE_W

    # Compute effect size
    effect_size_B_z_based = z_B / math.sqrt(n_B)
    print(f"Effect size r_B - z-based: {effect_size_B_z_based}")

    return


# This function makes sense
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

    # Final check: Print the number of tokenized sentences
    print(f"Number of sentences successfully tokenized: {len(input_ids)}")
    return input_ids


# This function makes sense
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
                              value=tokenizer.mask_token_id,
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


# This function makes sense
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
        # print("Input IDs for first document:", input_ids_TAM[0])
        # print("Decoded tokens:", tokenizer.convert_ids_to_tokens(input_ids_TAM[0]))

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
        print(f"Prior probability (p_prior): {prior}")

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

    # print('max_len evaluation: {}'.format(max_len_eval))

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


# TAKEN FROM TUTORIAL

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# Helper function for formatting elapsed times.
def format_time(elapsed):
    """ Takes a time in seconds and returns a string hh:mm:ss"""
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# COPY END
