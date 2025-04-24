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
![Screenshot 2025-04-01 215825](https://github.com/user-attachments/assets/b39204d1-334a-42cc-a8eb-347f31c749b3)
![Screenshot 2025-04-01 224045](https://github.com/user-attachments/assets/b8e28af4-3418-4dc6-9b10-d159512a2233)



---
