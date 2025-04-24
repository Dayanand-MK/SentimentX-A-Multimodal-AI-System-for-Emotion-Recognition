# 🎭 SentimentX: A Multimodal AI System for Emotion Recognition

**SentimentX** is a cutting-edge AI system that understands human emotions through **video**, **audio**, and **text**. It uses powerful deep learning models like CNNs, Transformers, and BERT, and wraps everything in a sleek and user-friendly **Tkinter GUI**.



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
├── Main.py                      # 🧠 Main GUI application
├── Execution.txt                # ✅ Process flow checklist

├── video_analysis/              # 🎥 Video Emotion Recognition
│   ├── sentivideo.py            # Real-time webcam emotion detection
│   ├── deep_emotion.py          # CNN + Spatial Transformer Network model
│   ├── train_emotion_model.py   # Training script for CNN model
│   ├── data_load.py             # Custom dataset class for image loading
│   ├── prepare_dataset.py       # Dataset preprocessor (CSV generator)
│   └── deep_emotion-*.pt        # Trained model file (weights)

├── audio_analysis/              # 🎙️ Audio Sentiment Analysis
│   ├── torchscript.py           # Converts model to TorchScript (.pt)
│   ├── emotion_model.pth        # Pretrained PyTorch model

├── text_analysis/               # 📝 Text Sentiment Analysis
│   ├── sentitext.py             # DistilBERT model training & evaluation
│   ├── sentitext_load.py        # BERT model loader for inference
│   ├── sentimentds.csv          # Sample sentiment dataset
│   └── bert_sentiment_model/    # Saved BERT model + label_map.json

├── requirements.txt             # 📦 Python dependencies list
├── README.md                    # 📘 Project documentation
└── LICENSE                      # ⚖️ MIT License


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

## 📌 Screenshot
![Screenshot 2025-04-01 215227](https://github.com/user-attachments/assets/06cc0b35-0db2-4dc2-a6e1-05214e9f2f1a)
![Screenshot 2025-04-01 215806](https://github.com/user-attachments/assets/31edfbfd-f7c8-4117-824f-9222daaa3cde)
![Screenshot 2025-04-01 224442](https://github.com/user-attachments/assets/2cbc8dbe-dd16-4d46-ab39-d8e7398f807e)
![Screenshot 2025-04-01 224045](https://github.com/user-attachments/assets/b8e28af4-3418-4dc6-9b10-d159512a2233)

---
