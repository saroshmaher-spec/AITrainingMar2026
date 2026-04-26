from transformers import AutoTokenizer
sentence = "I love programming in Python!"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize(sentence)
print(tokens)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)
encoded_input = tokenizer(sentence, return_tensors='pt')
print(encoded_input)
decoded_sentence = tokenizer.decode(token_ids)
print(decoded_sentence) 