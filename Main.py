import tkinter as tk
from sentivideo import analyze_video  
from text_model import predict_sentiment
from tkinter import filedialog, messagebox
import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame
import webbrowser

root = tk.Tk()
root.title("Sentiment Analysis")
root.attributes("-fullscreen", True)
root.configure(bg="#1e1e1e")  

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)

def open_text_analysis():
    text_window = tk.Toplevel(root)
    text_window.title("Text Analysis")
    text_window.geometry("500x400")
    text_window.configure(bg="#1e1e1e")
    
    tk.Label(text_window, text="Enter Text:", font=("Arial", 14), fg="white", bg="#1e1e1e").pack(pady=10)
    text_entry = tk.Entry(text_window, width=50, font=("Arial", 14))
    text_entry.pack(pady=10)
    
    result_label = tk.Label(text_window, text="", font=("Arial", 16, "bold"), fg="#00ffcc", bg="#1e1e1e")
    result_label.pack(pady=20)
    
    def on_predict():
        user_text = text_entry.get()
        if user_text.strip():
            sentiment = predict_sentiment(user_text)
            result_label.config(text=f"Predicted Sentiment: {sentiment}")
    
    predict_button = tk.Button(text_window, text="Predict Sentiment", font=("Arial", 14), fg="white", bg="#444", command=on_predict)
    predict_button.pack(pady=10)

def open_about_us():
    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Set window size to 80% of screen size
    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.7)

    about_window = tk.Toplevel(root)
    about_window.title("About Us")
    about_window.geometry(f"{window_width}x{window_height}")
    about_window.configure(bg="#1e1e1e")

    # Create a frame for better content organization
    frame = tk.Frame(about_window, bg="#1e1e1e")
    frame.pack(fill="both", expand=True, padx=30, pady=30)

    # Title label with a more modern look
    title_label = tk.Label(frame, text="About Us", font=("Helvetica", 28, "bold"), fg="#00ffcc", bg="#1e1e1e")
    title_label.pack(pady=20)

    # Detailed about text with better formatting
    about_text = (
        
        "👨‍💻 Developed by:\n"
        "- Dayanand M K (Lead Developer)\n"
        "- Balaji S\n"
        "- Praveen K S\n\n"

        "🌟 Who are we ?:\n"
        "We are 'Batch 9zX' of Computer Science and Engineering (Artificial Intelligence And Machine Learning)\n"
        "We are 2nd year students (4th semester)\n\n"

        "🌟 Our Goal:\n"
        "We aspire to revolutionize how AI systems understand and react to human emotions, making it more accessible for industries like healthcare, marketing, and entertainment.\n\n"

        "🔮 Future Plans:\n"
        "- Expand emotion recognition to include more nuanced emotional states.\n"
        "- Improve the multimodal system for even better real-time performance.\n"
        "- Integrate with VR/AR for a truly immersive emotional experience."
    )

    # Adding a cool, modern design for the content
    about_label = tk.Label(frame, text=about_text, font=("Helvetica", 14), fg="white", bg="#1e1e1e", justify="left", wraplength=window_width * 0.75)
    about_label.pack(pady=20)

    # Add a footer with some additional styling
    footer_label = tk.Label(frame, text="SentimentX - Pioneering Emotion-Aware AI", font=("Arial", 12, "italic"), fg="#00ffcc", bg="#1e1e1e")
    footer_label.pack(side="bottom", pady=20)

def open_pdf():
    webbrowser.open(r"E:\SentimentAnalysisSoftware\Sentiment analysis paper1.pdf") 

## Audio

class ParallelModel(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        self.conv2Dblock = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
            nn.Dropout(0.5)
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

EMOTIONS = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 0: 'surprise'}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ParallelModel(num_emotions=len(EMOTIONS)).to(device)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()

def get_mel_spectrogram(audio, sample_rate=16000):
    return librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=1024, hop_length=256, n_mels=128, fmax=sample_rate/2, power=2.0)

def preprocess_audio(file_path, sample_rate=16000, duration=3):
    audio, _ = librosa.load(file_path, sr=sample_rate, duration=duration, offset=0.5)
    signal = np.zeros(int(sample_rate * duration), dtype=np.float32)
    signal[:len(audio)] = audio.astype(np.float32)
    mel_spec = get_mel_spectrogram(signal, sample_rate)
    return torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

def play_audio(file_path):
    if not file_path:
        messagebox.showwarning("Warning", "No file selected!")
        return
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    except Exception as e:
        messagebox.showerror("Error", f"Could not play audio: {str(e)}")


        
def open_audio_analysis():
    audio_window = tk.Toplevel(root)
    audio_window.title("Audio Analysis")
    audio_window.geometry("500x400")
    audio_window.configure(bg="#1e1e1e")

    tk.Label(audio_window, text="Upload an Audio File:", font=("Arial", 14), fg="white", bg="#1e1e1e").pack(pady=10)
    
    file_path_var = tk.StringVar()

    def select_file():
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        file_path_var.set(file_path)
    
    tk.Button(audio_window, text="Browse", font=("Arial", 12), command=select_file, bg="#444", fg="white").pack(pady=10)
    tk.Label(audio_window, textvariable=file_path_var, font=("Arial", 12), fg="white", bg="#1e1e1e").pack(pady=5)

    result_label = tk.Label(audio_window, text="", font=("Arial", 16, "bold"), fg="#00ffcc", bg="#1e1e1e")
    result_label.pack(pady=20)

    def predict_audio():
        file_path = file_path_var.get()
        if not file_path:
            messagebox.showwarning("Warning", "Please select an audio file!")
            return
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "File not found!")
            return

        mel_spectrogram = preprocess_audio(file_path)
        with torch.no_grad():
            logits = model(mel_spectrogram)
            prediction = torch.argmax(logits, dim=1).item()
            sentiment = EMOTIONS[prediction]
        
        result_label.config(text=f"Predicted Emotion: {sentiment}")

    predict_button = tk.Button(audio_window, text="Analyze Audio", font=("Arial", 14), fg="white", bg="#444", command=predict_audio)
    predict_button.pack(pady=10)

    play_button = tk.Button(audio_window, text="🔊 Play Audio", font=("Arial", 14), fg="white", bg="blue", command=lambda: play_audio(file_path_var.get()))
    play_button.pack(pady=10)


label = tk.Label(root, text="Sentiment Analysis System", font=("Helvetica", 50, "bold"), fg="#00ffcc", bg="#1e1e1e")
label.grid(row=0, column=0, columnspan=2, pady=50)  

btn_style = {
    "font": ("Arial", 20, "bold"),
    "fg": "white",
    "bg": "#444",
    "activebackground": "#00ffcc",
    "activeforeground": "black",
    "width": 20,
    "height": 2,
    "bd": 5,
    "relief": "raised"
}

btn_video = tk.Button(root, text="🎥️ Video Analysis", command=analyze_video, **btn_style)
btn_video.grid(row=1, column=0, padx=20, pady=20)

btn_audio = tk.Button(root, text="🎙️ Audio Analysis", command=open_audio_analysis, **btn_style)
btn_audio.grid(row=1, column=1, padx=20, pady=20)

btn_text = tk.Button(root, text="📝️ Text Analysis", command=open_text_analysis,**btn_style)
btn_text.grid(row=2, column=0, padx=20, pady=20)

btn_about = tk.Button(root, text="ℹ️ About Us", command=open_about_us,**btn_style)
btn_about.grid(row=2, column=1, padx=20, pady=20)

btn_paper = tk.Button(root, text="📚️ Research Paper", command=open_pdf,**btn_style)
btn_paper.grid(row=3, column=0, columnspan=2, padx=20, pady=20)  

btn_exit_style = btn_style.copy()
btn_exit_style.update({"bg": "red", "activebackground": "#ff4444"})

btn_exit = tk.Button(root, text="❌️️ Exit", command=root.quit, **btn_exit_style)
btn_exit.grid(row=4, column=0, columnspan=2, pady=50)  

root.mainloop()
