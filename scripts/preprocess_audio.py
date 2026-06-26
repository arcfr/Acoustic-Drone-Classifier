import os
import numpy as np
import librosa
import soundfile as sf

SOURCE_DIR = "./data/audiosamples"
TARGET_DIR = "./data_preprocessed/audio_preprocessed"
TARGET_SR = 22050

CLASSES = ["light", "medium", "heavy", "nodrone"]

def run_audio_preprocessing():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    print("Initiating Audio Signal Preprocessing Pipeline...")
    print(f"Target Specifications: Unified {TARGET_SR}Hz Rate | Peak Amplitude Normalization (1.0)\n")

    for c in CLASSES:
        source_folder = os.path.join(SOURCE_DIR, c)
        target_folder = os.path.join(TARGET_DIR, c)
        
        if not os.path.exists(source_folder):
            print(f"Warning: Subfolder '{c}' missing from source tree. Skipping.")
            continue
            
        os.makedirs(target_folder, exist_ok=True)
        audio_files = [f for f in os.listdir(source_folder) if f.endswith(".wav")]
        print(f"Cleaning [{c.upper()}]: Processing {len(audio_files)} files...")

        for file_name in audio_files:
            source_path = os.path.join(source_folder, file_name)
            target_path = os.path.join(target_folder, file_name)

            try:
                # 1. Unified Resampling during file load
                y, sr = librosa.load(source_path, sr=TARGET_SR)
                
                if len(y) == 0:
                    continue

                # 2. Amplitude Peak Normalization
                peak = np.max(np.abs(y))
                if peak > 0:
                    y = y / peak

                # Save the standardized waveform to the new target directory
                sf.write(target_path, y, TARGET_SR, format='WAV', subtype='PCM_16')

            except Exception as e:
                print(f"Error preprocessing sample {file_name}: {e}")

    print(f"\nPreprocessing Complete. Cleaned audio saved to: {os.path.abspath(TARGET_DIR)}")

if __name__ == "__main__":
    run_audio_preprocessing()