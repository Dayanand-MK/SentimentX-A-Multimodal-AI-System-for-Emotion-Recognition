import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

model_path = "./distilbert_sentiment_model"  
model = DistilBertForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

sentiment_labels = ["anger", "disgust", "fear", "happy", "sadness", "surprise"]

def predict_sentiment(text):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)
        predicted_label = torch.argmax(output.logits, dim=1).item()
    return sentiment_labels[predicted_label]

#sample_text = "I am very happy today!"
#print("Predicted Sentiment:", predict_sentiment(sample_text))
