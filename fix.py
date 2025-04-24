import numpy as np
import pandas as pd
import os
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

EMOTIONS = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 0: 'surprise'}
DATA_PATH = "E:/Ravdees_model/Ravdess_Dataset"
SAMPLE_RATE = 16000
DURATION = 3
BATCH_SIZE = 32
EPOCHS = 20  
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset
data_list = []
for dirname, _, filenames in os.walk(DATA_PATH):
    for filename in filenames:
        split_emo = filename.split('.')[0].split('-')
        emotion = int(split_emo[2])
        if emotion == 8:
            emotion = 0
        gender = 'female' if int(split_emo[6]) % 2 == 0 else 'male'
        data_list.append({"Emotion": emotion, "Gender": gender, "Path": os.path.join(dirname, filename)})

data = pd.DataFrame(data_list)

max_count = data["Emotion"].value_counts().max()
data = data.groupby("Emotion").apply(lambda x: x.sample(max_count, replace=True)).reset_index(drop=True)

signals = []
for file_path in tqdm(data.Path, desc='Loading Audio'):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, offset=0.5)
    signal = np.zeros(int(SAMPLE_RATE * DURATION), dtype=np.float32)
    signal[:len(audio)] = audio.astype(np.float32)
    signals.append(signal)

signals = np.stack(signals, axis=0)
labels = data["Emotion"].values

X_train, X_temp, Y_train, Y_temp = train_test_split(signals, labels, test_size=0.2, stratify=labels, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, stratify=Y_temp, random_state=42)

def getMELspectrogram(audio, sample_rate):
    return librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=1024, hop_length=256, n_mels=128, fmax=sample_rate/2, power=2.0)

def process_mel_spectrograms(X_data):
    return np.stack([getMELspectrogram(audio, SAMPLE_RATE).astype(np.float32) for audio in tqdm(X_data, desc='Mel Spectrograms')], axis=0)

X_train = process_mel_spectrograms(X_train)
X_val = process_mel_spectrograms(X_val)
X_test = process_mel_spectrograms(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class ParallelModel(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        self.conv2Dblock = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
            nn.Dropout(0.5)  # Added dropout to prevent overfitting
        )
        self.transf_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=128, nhead=4), num_layers=4)
        self.fc = nn.Linear(192, num_emotions)

    def forward(self, x):
        conv_embedding = torch.flatten(self.conv2Dblock(x), start_dim=1)
        x_reduced = x.squeeze(1).permute(2, 0, 1)
        transf_out = self.transf_encoder(x_reduced)
        transf_embedding = torch.mean(transf_out, dim=0)
        complete_embedding = torch.cat([conv_embedding, transf_embedding], dim=1)
        return self.fc(complete_embedding)

model = ParallelModel(num_emotions=len(EMOTIONS)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
loss_fnc = nn.CrossEntropyLoss()

best_val_loss = float('inf')
early_stopping_counter = 0
print("Training model...")
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_acc = 0, 0
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fnc(logits, Y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (torch.argmax(logits, dim=1) == Y_batch).float().mean().item()
    train_loss /= len(train_loader)
    train_acc = train_acc / len(train_loader) * 100
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for X_val_batch, Y_val_batch in val_loader:
            X_val_batch, Y_val_batch = X_val_batch.to(device), Y_val_batch.to(device)
            logits = model(X_val_batch)
            loss = loss_fnc(logits, Y_val_batch)
            val_loss += loss.item()
            val_acc += (torch.argmax(logits, dim=1) == Y_val_batch).float().mean().item()
    val_loss /= len(val_loader)
    val_acc = val_acc / len(val_loader) * 100
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")

torch.save(model.state_dict(), "emotion_model.pth")
print("Model saved as emotion_model.pth")