from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

sentence = "Sie arbeitet als Ärztin."
tokens = tokenizer.tokenize(sentence)
print(tokens)

sentence = "Sie ist eine Kindergarten Erzieherin."
tokens = tokenizer.tokenize(sentence)
print(tokens)

token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(list(zip(tokens, token_ids)))
