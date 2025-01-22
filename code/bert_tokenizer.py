from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

sentence = "Er ist ein Schreiner."
tokens = tokenizer.tokenize(sentence)
print(tokens)

sentence = "Sie ist eine Schreinerin."
tokens = tokenizer.tokenize(sentence)
print(tokens)

token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(list(zip(tokens, token_ids)))
