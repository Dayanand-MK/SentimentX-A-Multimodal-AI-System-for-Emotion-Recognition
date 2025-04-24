# 🎭 SentimentX: A Multimodal AI System for Emotion Recognition

**SentimentX** is a cutting-edge AI system that understands human emotions through **video**, **audio**, and **text**. It uses powerful deep learning models like CNNs, Transformers, and BERT, and wraps everything in a sleek and user-friendly **Tkinter GUI**.

![SentimentX Banner](https://via.placeholder.com/800x200.png?text=SentimentX+-+Emotion+Recognition+System)

---

## 💡 Key Features

- 🎥 **Video Emotion Detection**  
  Real-time facial emotion recognition using CNN + Spatial Transformer Network.

- 🎙️ **Audio Sentiment Analysis**  
  Emotion prediction from voice using a hybrid Transformer-CNN model.

- 📝 **Text Sentiment Analysis**  
  Sentiment classification using BERT/DistilBERT fine-tuned models.

- 🧠 **Unified GUI**  
  A full-screen Tkinter interface to interact with all three modalities.

- 🔊 Voice response with `pyttsx3`  
  Emotion results are announced via TTS for video predictions.

---

## 🗂️ Project Structure


SentimentX/
│
├── Main.py                     # 🧠 GUI entry point
├── Execution.txt               # ✅ Ordered process flow
│
├── video_analysis/
│   ├── sentivideo.py           # Video emotion detection logic
│   ├── deep_emotion.py         # CNN + STN model
│   ├── train_emotion_model.py  # Model training script
│   ├── data_load.py            # Custom PyTorch dataset
│   ├── prepare_dataset.py      # Dataset preprocessing (CSV)
│   └── deep_emotion-100-128-0.005.pt
│
├── audio_analysis/
│   ├── torchscript.py          # Converts model to TorchScript
│   ├── emotion_model.pth       # Pretrained model for audio emotion
│
├── text_analysis/
│   ├── sentitext.py            # Training + prediction using DistilBERT
│   ├── sentitext_load.py       # Model loader (BERT version)
│   ├── bert_sentiment_model/   # Saved BERT model and label_map.json
│   └── sentimentds.csv         # Sample dataset


---

## ▶️ Execution Flow

To set up and run this project, follow these steps **in order**:

1. `prepare_dataset.py`  
   ➤ Converts CK+ dataset folders into `Train1.csv` and `Test_1.csv`.

2. `train_emotion_model.py`  
   ➤ Trains and saves the CNN model (`deep_emotion-100-128-0.005.pt`).

3. `deep_emotion.py`  
   ➤ Contains model architecture used in training and inference.

4. `data_load.py`  
   ➤ Provides dataset class used by PyTorch's DataLoader.

5. `sentivideo.py`  
   ➤ Starts webcam-based real-time facial emotion recognition.

6. `Main.py`  
   ➤ Runs the GUI combining all three emotion detection modules.

---

## ⚙️ Installation

### 1. Clone this repo

git clone https://github.com/<your-username>/SentimentX.git
cd SentimentX


### 2. Install dependencies

pip install -r requirements.txt


You may also need:

pip install torch torchvision transformers scikit-learn opencv-python librosa pygame


### 3. Model Placement

| Model | Location |
|-------|----------|
| `deep_emotion-100-128-0.005.pt` | `video_analysis/` |
| `emotion_model.pth`             | `audio_analysis/` |
| `bert_sentiment_model/` folder  | `text_analysis/` |

---

## 🚀 Run the Application


python Main.py


---

## 🧑‍💻 Developers

👨‍💻 **Dayanand M K** — *Lead Developer*  
👨‍💻 Balaji S  
👨‍💻 Praveen K S  

Batch: 9 — 2nd Year CSE (AI & ML)  
Vel Tech High Tech Dr. Rangarajan Dr. Sakunthala Engineering College

---

## 🔮 Future Scope

- Fuse results from all three modalities to make smarter emotional decisions  
- Add support for complex, subtle emotions  
- Build VR/AR integrations for immersive feedback

---

## 📄 License

MIT License © 2025 Dayanand M K and Team

---

## 📌 Screenshot

![GUI Screenshot](https://via.placeholder.com/800x500.png?text=SentimentX+GUI)

---
