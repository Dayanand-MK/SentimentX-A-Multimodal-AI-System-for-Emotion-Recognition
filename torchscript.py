import torch
from fix import ParallelModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ParallelModel(num_emotions=len(EMOTIONS)).to(device)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()

# Convert to TorchScript
example_input = torch.randn(1, 1, 128, 188).to(device)  # Shape of a Mel spectrogram
traced_model = torch.jit.trace(model, example_input)

# Save as .pt file
traced_model.save("emotion_model.pt")
print("Model saved as emotion_model.pt")
