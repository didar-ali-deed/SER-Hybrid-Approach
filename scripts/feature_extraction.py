import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm  # For progress bar
import matplotlib.pyplot as plt

# Load the combined dataset
data_path = "../Preprocessed Data/combined_emotions.csv"
output_path = "../Extracted Features/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load the CSV with emotion labels and file paths
df = pd.read_csv(data_path)

# Function to extract MFCC, Chroma, and Mel Spectrogram features
def extract_features(file_path):
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(file_path, sr=16000)
        
        # Extract MFCC features (13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        # Extract Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)

        # Extract Mel Spectrogram features
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        mel_mean = np.mean(mel.T, axis=0)

        # Concatenate all features into one vector
        features = np.hstack([mfccs_mean, chroma_mean, mel_mean])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Iterate over the dataset and extract features
def process_audio_files(df):
    features = []
    labels = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting features"):
        file_path = row['Path']
        emotion = row['Emotions']
        feature_vector = extract_features(file_path)

        if feature_vector is not None:
            features.append(feature_vector)
            labels.append(emotion)

    return np.array(features), np.array(labels)

if __name__ == "__main__":
    # Extract features
    print("Starting feature extraction...")
    features, labels = process_audio_files(df)

    # Save extracted features to a CSV file
    feature_df = pd.DataFrame(features)
    feature_df['label'] = labels
    feature_df.to_csv(os.path.join(output_path, "audio_features.csv"), index=False)
    
    print(f"Features saved to {os.path.join(output_path, 'audio_features.csv')}")
