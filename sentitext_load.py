import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer from the saved directory
    model = BertForSequenceClassification.from_pretrained("bert_sentiment_model")
    tokenizer = BertTokenizer.from_pretrained("bert_sentiment_model")

    # Load label map
    with open("bert_sentiment_model/label_map.json", "r") as f:
        label_map = json.load(f)

    model.to(device)
    model.eval()
    
    return model, tokenizer, label_map

def predict_sentiment(text, model, tokenizer, label_map):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model(**inputs)
        predicted_label = torch.argmax(output.logits, dim=1).item()

    sentiment_labels = {v: k for k, v in label_map.items()}  # Reverse mapping
    return sentiment_labels[predicted_label]
