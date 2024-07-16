import os
import numpy as np
import torch
import torchaudio
import requests
import zipfile
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from config import config

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

def download_ravdess():
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
    dataset_path = os.path.join(config.DATA_DIR, "RAVDESS_speech.zip")

    if not os.path.exists(dataset_path):
        print("Downloading dataset...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(dataset_path, 'wb') as f:
                f.write(response.content)
            print("Download complete.")
        else:
            print("Failed to download the dataset.")
            return config.DATA_DIR, False
    else:
        print("Dataset already exists. Skipping download.\n")

    extracted_path = os.path.join(config.DATA_DIR, "RAVDESS_speech")
    if not os.path.exists(extracted_path):
        try:
            with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_path)
            print("Extracted dataset.")
        except Exception as e:
            print(f"Extraction failed: {e}")
            return config.DATA_DIR, False
    else:
        print("Dataset already extracted.\n")
    return extracted_path, True

def preprocess_data(data_dir):
    data = []
    labels = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                label = int(file.split('-')[2]) - 1
                data.append(file_path)
                labels.append(label)
    if len(data) == 0:
        raise ValueError("No valid .wav files found in the dataset.")
    return np.array(data), np.array(labels)

def extract_features(waveform, sample_rate):
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec2_model(**inputs)
    wav2vec2_features = outputs.last_hidden_state.squeeze(0).mean(dim=0).numpy()
    return wav2vec2_features.reshape(1, -1)  # reshape (1, input_size) 
  
class RAVDESSTorchDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data[idx]
        label = self.labels[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        features = extract_features(waveform, sample_rate)
        return features, label

def collate_batch(batch):
    features, labels = zip(*batch)
    features_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(f, dtype=torch.float32) for f in features], batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return features_padded, labels

def prepare_dataloaders(data, labels, BATCH_SIZE):
    full_dataset = RAVDESSTorchDataset(data, labels)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    
    return train_loader, val_loader, test_loader