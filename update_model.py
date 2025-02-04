import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# Load dataset
df = pd.read_csv("data_train.csv")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Ensure column names are correct
X = df["text"].tolist()
y = df["label"].tolist()

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare datasets
train_dataset = TextDataset(X_train, y_train, tokenizer)
val_dataset = TextDataset(X_val, y_val, tokenizer)

# Save preprocessed data
torch.save(train_dataset, "train_data.pt")
torch.save(val_dataset, "val_data.pt")
