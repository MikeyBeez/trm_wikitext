from datasets import load_dataset

# Load WikiText-103 train and validation splits
dataset = load_dataset("wikitext", "wikitext-103-v1")

train_texts = dataset["train"]["text"]
val_texts = dataset["validation"]["text"]

# Concatenate into single strings if needed
train_data = "\n".join(train_texts)
val_data = "\n".join(val_texts)

