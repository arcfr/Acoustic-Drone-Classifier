import os
import numpy as np
import librosa

BASE_DATA_DIR = "./data_preprocessed/audio_preprocessed"
OUTPUT_DIR = "./data_preprocessed/log_mel_spectograms"

LABEL_MAPPING = {
    "light": 0,
    "medium": 1,
    "heavy": 2,
    "nodrone": 3
}

# ResNet-friendly target dimensions
N_MELS = 128       # Height of the image
TARGET_WIDTH = 87  # Width of the image (time steps)

def extract_spectrogram_pipeline():
    if not os.path.exists(BASE_DATA_DIR):
        print(f"Error: Base data folder '{BASE_DATA_DIR}' not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    X_data = []
    y_data = []
    
    print("Extracting Log-Mel Spectrograms for Transfer Learning...\n")
    
    for category, label_int in LABEL_MAPPING.items():
        cat_folder = os.path.join(BASE_DATA_DIR, category)
        
        if not os.path.exists(cat_folder):
            print(f"Warning: Subfolder '{category}' missing. Skipping.")
            continue
            
        audio_files = [f for f in os.listdir(cat_folder) if f.endswith(".wav")]
        print(f"Processing [{category.upper()}]: {len(audio_files)} files.")
        
        for file_name in audio_files:
            file_path = os.path.join(cat_folder, file_name)
            
            try:
                y, sr = librosa.load(file_path, sr=22050)
                if len(y) == 0:
                    continue
                
                # 1. Compute Raw Mel Spectrogram
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
                
                # 2. Convert to Log Scale (Decibels)
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                
                # 3. Handle Time Dimension Padding/Trimming
                if log_mel_spec.shape[1] < TARGET_WIDTH:
                    pad_width = TARGET_WIDTH - log_mel_spec.shape[1]
                    log_mel_spec = np.pad(log_mel_spec, pad_width=((0, 0), (0, pad_width)), mode='constant')
                elif log_mel_spec.shape[1] > TARGET_WIDTH:
                    log_mel_spec = log_mel_spec[:, :TARGET_WIDTH]
                
                # 4. Duplicate grayscale spectrogram across 3 channels to simulate RGB
                rgb_spectrogram = np.stack([log_mel_spec, log_mel_spec, log_mel_spec], axis=-1)
                
                X_data.append(rgb_spectrogram)
                y_data.append(label_int)
                
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int64)
    
    # Global normalization scale
    mean = np.mean(X)
    std = np.std(X)
    if std > 0:
        X = (X - mean) / std
    np.save(os.path.join(OUTPUT_DIR, "X_resnet.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_resnet.npy"), y)
    
    print("\nSpectrogram Matrix Compilation Complete.")
    print(f"Saved to {OUTPUT_DIR} with shape: {X.shape}")

if __name__ == "__main__":
    extract_spectrogram_pipeline()