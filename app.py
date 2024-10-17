from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import pandas as pd

app = Flask(__name__)

# Define the model architecture
class SoundCNN(nn.Module):
    def __init__(self):
        super(SoundCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64 * 1 * 1, 128)
        self.fc2 = nn.Linear(128, 50)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Reshape x to [batch_size, 64*1*1]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the pre-trained model and weights
model = SoundCNN()
model.load_state_dict(torch.load('final_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Load the metadata (for decoding labels)
metadata = pd.read_csv('archive-3/esc50.csv')
decoder = dict(zip(metadata['target'], metadata['category']))

# Helper functions
def pad_or_truncate_spectrogram(spectrogram, max_frames):
    if spectrogram.shape[1] < max_frames:
        pad_width = max_frames - spectrogram.shape[1]
        padded_spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        padded_spectrogram = spectrogram[:, :max_frames]
    return padded_spectrogram

def extract_features(data, sr, max_frames=512):
    mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram_db = pad_or_truncate_spectrogram(mel_spectrogram_db, max_frames)
    mean = np.mean(mel_spectrogram_db)
    std = np.std(mel_spectrogram_db)
    mel_spectrogram_db = (mel_spectrogram_db - mean) / std
    return mel_spectrogram_db

# Flask route for audio prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    # Get the audio file from the request
    file = request.files['file']
    
    # Load and preprocess the audio
    try:
        audio_data, sr = librosa.load(file, sr=22050)
        features = extract_features(audio_data, sr)
        features = features.reshape(1, 1, features.shape[0], features.shape[1])
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Perform inference
        with torch.no_grad():
            output = model(features_tensor)
            _, predicted = torch.max(output, 1)
        
        # Get the predicted category
        predicted_category = decoder[predicted.item()]
        
        return jsonify({'predicted_category': predicted_category})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


# curl -X POST -F 'file=@coding-fast-typing-on-keyboard-sound-247411.mp3' http://127.0.0.1:5000/predict
