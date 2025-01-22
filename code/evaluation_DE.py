from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn.functional as F

device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
model = AutoModelForMaskedLM.from_pretrained("dbmdz/bert-base-german-cased")
print("loading german bert")

text = "Er ist Kindergärtner"

text_mask = "[MASK], die Kindergärtnerin, hatte einen guten Arbeitstag."

inputs = tokenizer(text_mask, return_tensors="pt")

#print(inputs)

outputs = model(**inputs)
logits = outputs.logits


# Find the index of the [MASK] token
mask_token_indices = (inputs.input_ids ==
                      tokenizer.mask_token_id).nonzero(as_tuple=True)[1]


# Get the token ID for the original word "he"
original_token_id = tokenizer.convert_tokens_to_ids("Er")

# Loop over each [MASK] token and predict its top tokens with probabilities
# Loop over each [MASK] token and get probabilities for all tokens
for idx in mask_token_indices[:1]:
    # Extract logits for the current [MASK] token
    mask_token_logits = logits[0, idx, :]

    # Convert logits to probabilities
    probs = F.softmax(mask_token_logits, dim=-1)

    # Get the top 5 token IDs and their probabilities
    top_5_probs, top_5_indices = torch.topk(probs, 5)

    # Get the probability of the original token
    original_token_prob = probs[original_token_id].item()

    # Map indices to tokens and print their probabilities
    print(f"Top 5 predictions for [MASK] token at index {idx.item()}:")
    for i in range(5):
        token = tokenizer.convert_ids_to_tokens(top_5_indices[i].item())
        probability = top_5_probs[i].item()
        print(f"{token}: {probability:.6f}")

print(f"Probability of the original token 'sie': {original_token_prob:.4f}")

