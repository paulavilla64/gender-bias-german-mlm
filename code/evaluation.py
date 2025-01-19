from transformers import BertTokenizer, BertForMaskedLM
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

def get_word_at_mask_position(text, text_mask, model_name="bert-base-uncased"):
    """
    Returns the word from the original text that corresponds to the [MASK] position in the text_mask.

    Args:
        text (str): The original sentence (e.g., "he is a taper").
        text_mask (str): The sentence with [MASK] (e.g., "[MASK] is a taper").
        model_name (str): The name of the tokenizer model to use (default: "bert-base-uncased").

    Returns:
        str: The word from the original text corresponding to the [MASK] token.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize both text and text_mask
    text_tokens = tokenizer.tokenize(text)
    text_mask_tokens = tokenizer.tokenize(text_mask)

    # Find the position of [MASK] in the tokenized text_mask
    try:
        mask_index = text_mask_tokens.index(tokenizer.mask_token)  # Use tokenizer's [MASK] token
    except ValueError:
        raise ValueError("[MASK] token not found in text_mask")

    # Return the word at the same position from the original text
    if mask_index < len(text_tokens):
        return tokenizer.convert_tokens_to_string([text_tokens[mask_index]])
    else:
        raise ValueError("The [MASK] position does not align with the original text.")


device = torch.device("cpu")

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
print("loading bert")
path = "/mount/studenten-temp1/users/villavpa/thesis/gender-bias-BERT/code/checkpoints_epochs/epoch-3.pt"
model.load_state_dict(torch.load(path, map_location=device))
print("loading checkpoints")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "he is a taper"

text_mask = "[MASK] is a taper" 

inputs = tokenizer(text_mask, return_tensors="pt")

print(inputs)

outputs = model(**inputs)
logits = outputs.logits


# Find the index of the [MASK] token
mask_token_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]


# Get the token ID for the original word "he"
original_token_id = tokenizer.convert_tokens_to_ids("she")

# Loop over each [MASK] token and predict its top tokens with probabilities
for idx in mask_token_indices[:1]:
    # Extract logits for the current [MASK] token
    mask_token_logits = logits[0, idx, :]
    
    # Convert logits to probabilities
    probs = F.softmax(mask_token_logits, dim=-1)  

    # Get the probability of the original token
    original_token_prob = probs[original_token_id].item()

print(f"Probability of the original token 'she': {original_token_prob:.4f}")
