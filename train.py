from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, Dataset
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

# Load dataset from CSV
df = pd.read_csv("data/preprocessed.csv")

# Encode the 'tag' column with integer labels (since it's a classification task)
label_encoder = LabelEncoder()
df["tag"] = label_encoder.fit_transform(df["tag"])

# Convert to HuggingFace Dataset format
dataset = Dataset.from_pandas(df)

# Split dataset into train and validation sets
train_dataset = dataset.train_test_split(test_size=0.2)["train"]
val_dataset = dataset.train_test_split(test_size=0.2)["test"]

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)


# Tokenize the dataset and include the 'labels' as mood tags
def tokenize_function(examples):
    # Tokenize the 'input' column and include 'labels' from 'tag' column
    encodings = tokenizer(examples["input"], padding="max_length", truncation=True)
    encodings["labels"] = examples["tag"]
    return encodings


train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Load model for classification (number of moods as the number of labels)
model = BertForSequenceClassification.from_pretrained(
    model_name, num_labels=len(df["tag"].unique())
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Use eval_strategy to avoid future warnings
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer for future use
model.save_pretrained("./mood_classifier")
tokenizer.save_pretrained("./mood_classifier")
