import os
import pandas as pd

# Define paths to your datasets
RAVDESS = "../data/ravdess/"
CREMA = "../data/crema-d/AudioWAV/"
TESS = "../data/tess/TESS Toronto emotional speech set data/"
SAVEE = "../data/savee/ALL/"

# RAVDESS Dataset
def process_ravdess():
    ravdess_directory_list = os.listdir(RAVDESS)
    file_emotion = []
    file_path = []
    
    # Loop through each actor subdirectory (e.g., Actor_01, Actor_02, etc.)
    for actor_dir in ravdess_directory_list:
        actor_path = os.path.join(RAVDESS, actor_dir)
        
        if os.path.isdir(actor_path):
            actor_files = os.listdir(actor_path)
            
            for file in actor_files:
                if file.endswith(".wav"):
                    part = file.split('.')[0].split('-')
                    if len(part) >= 3:
                        emotion_code = int(part[2])
                        file_emotion.append(emotion_code)
                        file_path.append(os.path.join(actor_path, file))
                    else:
                        print(f"Skipping file with unexpected format: {file}")
    
    # Create dataframe for RAVDESS
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    ravdess_df = pd.concat([emotion_df, path_df], axis=1)
    
    # Replace emotion codes with emotion names (without inplace=True)
    ravdess_df['Emotions'] = ravdess_df['Emotions'].replace({
        1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 
        6: 'fear', 7: 'disgust', 8: 'surprise'
    })
    
    return ravdess_df

# CREMA-D Dataset
def process_crema():
    crema_directory_list = os.listdir(CREMA)
    file_emotion = []
    file_path = []

    for file in crema_directory_list:
        if file.endswith(".wav"):
            file_path.append(os.path.join(CREMA, file))
            part = file.split('_')
            emotion_code = part[2]
            if emotion_code == 'SAD':
                file_emotion.append('sad')
            elif emotion_code == 'ANG':
                file_emotion.append('angry')
            elif emotion_code == 'DIS':
                file_emotion.append('disgust')
            elif emotion_code == 'FEA':
                file_emotion.append('fear')
            elif emotion_code == 'HAP':
                file_emotion.append('happy')
            elif emotion_code == 'NEU':
                file_emotion.append('neutral')
            elif emotion_code == 'SUR':
                file_emotion.append('surprise')       
            else:
                file_emotion.append('Unknown')
    
    # Create dataframe for CREMA-D
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    crema_df = pd.concat([emotion_df, path_df], axis=1)
    
    return crema_df

# TESS Dataset
# TESS Dataset
def process_tess():
    tess_directory_list = os.listdir(TESS)
    file_emotion = []
    file_path = []

    # Loop through emotion folders (e.g., OAF_angry, YAF_happy, etc.)
    for dir in tess_directory_list:
        emotion_folder = os.path.join(TESS, dir)
        
        if os.path.isdir(emotion_folder):
            files = os.listdir(emotion_folder)
            for file in files:
                if file.endswith(".wav"):
                    # The emotion is in the folder name, so extract from the folder name
                    emotion = dir.split('_')[-1].lower()  # Extract emotion from folder name like 'angry', 'disgust', etc.
                    file_emotion.append(emotion)
                    file_path.append(os.path.join(emotion_folder, file))
    
    # Create dataframe for TESS
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    tess_df = pd.concat([emotion_df, path_df], axis=1)
    
    return tess_df

# SAVEE Dataset
def process_savee():
    savee_directory_list = os.listdir(SAVEE)
    file_emotion = []
    file_path = []

    for file in savee_directory_list:
        if file.endswith(".wav"):
            file_path.append(os.path.join(SAVEE, file))
            part = file.split('_')[1][:-6]  # Extract emotion prefix from filename
            if part == 'a':
                file_emotion.append('angry')
            elif part == 'd':
                file_emotion.append('disgust')
            elif part == 'f':
                file_emotion.append('fear')
            elif part == 'h':
                file_emotion.append('happy')
            elif part == 'n':
                file_emotion.append('neutral')
            elif part == 'sa':
                file_emotion.append('sad')
            else:
                file_emotion.append('surprise')
    
    # Create dataframe for SAVEE
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    savee_df = pd.concat([emotion_df, path_df], axis=1)
    
    return savee_df

if __name__ == "__main__":
    # Process each dataset
    ravdess_df = process_ravdess()
    crema_df = process_crema()
    tess_df = process_tess()
    savee_df = process_savee()

    # Combine all datasets into a single dataframe
    combined_df = pd.concat([ravdess_df, crema_df, tess_df, savee_df], axis=0)

    # Ensure the output directory exists
    output_dir = "../Preprocessed Data/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the combined dataframe to a CSV file
    combined_df.to_csv(os.path.join(output_dir, "combined_emotions.csv"), index=False)
    
    print(f"Combined dataset saved to {os.path.join(output_dir, 'combined_emotions.csv')}")
