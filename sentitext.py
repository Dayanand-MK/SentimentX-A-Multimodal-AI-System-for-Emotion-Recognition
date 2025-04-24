### Module 1: Data Preprocessing
import pandas as pd
import torch
#!pip install evaluate
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

file_path=""
# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    label_map = {"anger": 0, "disgust": 1, "fear": 2, "happy": 3, "sadness": 4, "surprise": 5}
    df["label"] = df["Sentiment"].map(label_map)
    return df

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

### Module 2: Data Preparation
df = load_dataset("sentimentds.csv")
X_train, X_val, y_train, y_val = train_test_split(df["Text"], df["label"], test_size=0.1, random_state=42)
train_dataset = SentimentDataset(X_train, y_train)
val_dataset = SentimentDataset(X_val, y_val)

### Module 3: DistilBERT Model Initialization
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    report_to="none"  # Disable wandb
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)


import evaluate

# Load accuracy metric
metric = evaluate.load("accuracy")


# Define accuracy metric
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    return metric.compute(predictions=predictions.numpy(), references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics  # Include accuracy calculation
)

### Model Training & Evaluation
trainer.train()

# Evaluate on validation set
eval_results = trainer.evaluate()
print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")

# Evaluate on training set
train_results = trainer.evaluate(train_dataset)
print(f"Training Accuracy: {train_results['eval_accuracy']:.4f}")


### Module 4: Model Training

model.save_pretrained("./distilbert_sentiment_model")


### Module 5: Prediction Loop
def predict_sentiment(text):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Ensure model is on the correct device

    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)  # Move inputs to the same device
    with torch.no_grad():
        output = model(**inputs)
        predicted_label = torch.argmax(output.logits, dim=1).item()
    sentiment_labels = ["anger", "disgust", "fear", "happy", "sadness", "surprise"]
    return sentiment_labels[predicted_label]

while True:
    user_input = input("Enter text to predict sentiment (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print("Predicted Sentiment:", predict_sentiment(user_input))



