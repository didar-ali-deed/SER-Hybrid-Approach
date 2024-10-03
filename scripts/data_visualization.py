import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display

# Path to save visualizations
save_dir = r"D:\1.UET PESHAWAR\7th Semester Data\SER_Project\Data Visualization"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load the extracted features
features_path = "../Extracted Features/audio_features.csv"
df = pd.read_csv(features_path)

# Distribution of Emotions
def plot_emotion_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='label', palette="Set2")
    plt.title("Distribution of Emotions")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()  # Ensure proper layout
    
    # Save the plot
    save_path = os.path.join(save_dir, "emotion_distribution.png")
    plt.savefig(save_path)
    print(f"Emotion distribution plot saved to {save_path}")
    plt.show()

# Plot Waveplot (Waveform) for an audio file
def plot_waveplot(audio_path, emotion):
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return

    y, sr = librosa.load(audio_path)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveplot for {emotion}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    # Save the waveplot
    save_path = os.path.join(save_dir, f"waveplot_{emotion}.png")
    plt.savefig(save_path)
    print(f"Waveplot for {emotion} saved to {save_path}")
    plt.show()

# Plot Spectrogram for an audio file
def plot_spectrogram(audio_path, emotion):
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return

    y, sr = librosa.load(audio_path)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    log_spectrogram = librosa.power_to_db(spectrogram)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram for {emotion}")
    plt.tight_layout()

    # Save the spectrogram
    save_path = os.path.join(save_dir, f"spectrogram_{emotion}.png")
    plt.savefig(save_path)
    print(f"Spectrogram for {emotion} saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # Plot emotion distribution
    plot_emotion_distribution(df)
    
    # Dictionary to map emotions to example file paths
    emotion_file_map = {
        "neutral": "../data/tess/TESS Toronto emotional speech set data/OAF_neutral/OAF_back_neutral.wav",
        "calm": "../data/crema-d/AudioWAV/1001_DFA_NEU_XX.wav",  # Make sure this file exists
        "happy": "../data/ravdess/Actor_01/03-01-03-01-01-01-01.wav",
        "sad": "../data/ravdess/Actor_01/03-01-04-01-01-01-01.wav",
        "angry": "../data/tess/TESS Toronto emotional speech set data/OAF_angry/OAF_back_angry.wav",
        "fear": "../data/tess/TESS Toronto emotional speech set data/OAF_fear/OAF_back_fear.wav",
        "disgust": "../data/tess/TESS Toronto emotional speech set data/OAF_disgust/OAF_back_disgust.wav",
        "surprise": "../data/tess/TESS Toronto emotional speech set data/OAF_pleasant_surprise/OAF_back_ps.wav"
    }

    # Loop through the emotion_file_map and plot waveplots and spectrograms for each emotion
    for emotion, audio_path in emotion_file_map.items():
        # Plot and save the waveplot
        plot_waveplot(audio_path, emotion)

        # Plot and save the spectrogram
        plot_spectrogram(audio_path, emotion)
