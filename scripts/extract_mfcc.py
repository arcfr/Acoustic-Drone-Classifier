import os
import numpy as np
import librosa

BASE_DATA_DIR = "./data/audiosamples"
OUTPUT_DIR = "./data/mfccfeatures"

LABEL_MAPPING = {
    "light": 0,
    "medium": 1,
    "heavy": 2,
    "nodrone": 3
}

N_MFCC = 24
TARGET_WIDTH = 87

def extract_features_pipeline():
    if not os.path.exists(BASE_DATA_DIR):
        print(f"Error: Base data folder '{BASE_DATA_DIR}' not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    X_data = []
    y_data = []
    
    print("Initiating Enhanced 3-Channel MFCC Extraction Pipeline...\n")
    
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
                    
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
                
                if mfcc.shape[1] < TARGET_WIDTH:
                    pad_width = TARGET_WIDTH - mfcc.shape[1]
                    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
                elif mfcc.shape[1] > TARGET_WIDTH:
                    mfcc = mfcc[:, :TARGET_WIDTH]
                
                delta = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)
                
                three_channel_block = np.stack([mfcc, delta, delta2], axis=-1)
                
                X_data.append(three_channel_block)
                y_data.append(label_int)
                
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int64)
    
    print("\nNormalizing feature layers globally...")
    mean = np.mean(X)
    std = np.std(X)
    if std > 0:
        X = (X - mean) / std
        
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    
    print("\nMatrix Compilation Complete.")
    print(f"Final Feature Set (X.npy) Shape: {X.shape}")
    print(f"Final Target Labels (y.npy) Shape: {y.shape}")

if __name__ == "__main__":
    extract_features_pipeline()