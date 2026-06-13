from transformers import AutoTokenizer

# Load pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Sample text description
text = "A beautiful red flower in a garden"

# Tokenize and encode
encoded = tokenizer(
    text,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print("Original Text:")
print(text)

print("\nInput IDs:")
print(encoded["input_ids"])

print("\nAttention Mask:")
print(encoded["attention_mask"])

print("\nTokenized Words:")
print(tokenizer.tokenize(text))
