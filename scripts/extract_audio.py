import os
import math
import json
from yt_dlp import YoutubeDL
from pydub import AudioSegment

CHUNK_DURATION = 2
BASE_DATA_DIR = "./data"
JSON_FILE = "scripts/video_metadata.json"
SILENCE_THRESHOLD_DB = -80 

def setup_directories():
    classes = ["light", "medium", "heavy"]
    for c in classes:
        target_path = os.path.join(BASE_DATA_DIR, c)
        os.makedirs(target_path, exist_ok=True)
    print(f"Directory verification complete: {os.path.abspath(BASE_DATA_DIR)}")

def timestamp_to_ms(ts_str):
    if not ts_str or ts_str == "inf":
        return None
    parts = list(map(int, ts_str.split(':')))
    if len(parts) == 3:
        return ((parts[0] * 3600) + (parts[1] * 60) + parts[2]) * 1000
    elif len(parts) == 2:
        return ((parts[0] * 60) + parts[1]) * 1000
    return int(ts_str) * 1000

def download_and_slice():
    if not os.path.exists(JSON_FILE):
        print(f"Error: Cannot find {JSON_FILE} in the current directory.")
        return
        
    with open(JSON_FILE, "r") as f:
        video_collection = json.load(f)

    setup_directories()
    total_videos = len(video_collection)
    class_counters = {"light": 1, "medium": 1, "heavy": 1}

    print(f"Starting extraction pipeline for {total_videos} videos with real-time silence stripping...\n")

    for idx, item in enumerate(video_collection):
        url = item['url']
        label = item['label']
        has_ts = item['has_timestamps']
        start_str = item['start_time']
        end_str = item['end_time']
        
        temp_raw_name = f"temp_raw_vid_{idx+1}"
        expected_download_path = f"{temp_raw_name}.wav"
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{temp_raw_name}.%(ext)s',
            'nocheckcertificate': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'quiet': True,
            'no_warnings': True,
        }

        print(f"[{idx+1}/{total_videos}] Processing [{label.upper()}] from: {url}")
        
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            if not os.path.exists(expected_download_path):
                print(f"Warning: Download failed for video {idx+1}. Skipping.")
                continue

            audio = AudioSegment.from_wav(expected_download_path)
            
            if has_ts:
                start_ms = timestamp_to_ms(start_str) or 0
                end_ms = timestamp_to_ms(end_str) or len(audio)
                audio = audio[start_ms:end_ms]

            chunk_ms = CHUNK_DURATION * 1000
            total_length_ms = len(audio)
            num_chunks = math.floor(total_length_ms / chunk_ms)
            
            target_folder = os.path.join(BASE_DATA_DIR, label)
            chunks_saved = 0
            chunks_skipped = 0

            for i in range(num_chunks):
                c_start = i * chunk_ms
                c_end = (i + 1) * chunk_ms
                chunk = audio[c_start:c_end]
                
                # Dynamic validation: Check if the segment decibel level is above our floor
                if chunk.dBFS < SILENCE_THRESHOLD_DB:
                    chunks_skipped += 1
                    continue # Drops the silent clip instantly without writing to disk
                
                current_id = class_counters[label]
                output_filename = f"{label}_{current_id}.wav"
                
                chunk.export(os.path.join(target_folder, output_filename), format="wav")
                chunks_saved += 1
                class_counters[label] += 1
                
            print(f"   Success: Saved {chunks_saved} strict segments. Skipped {chunks_skipped} silent zones.")
            os.remove(expected_download_path)

        except Exception as e:
            print(f"   Error while processing video {idx+1}. Details: {e}")
            if os.path.exists(expected_download_path):
                os.remove(expected_download_path)

    print("\nPipeline Finished successfully.")
    print(f"Total Light Samples: {class_counters['light'] - 1}")
    print(f"Total Medium Samples: {class_counters['medium'] - 1}")
    print(f"Total Heavy Samples: {class_counters['heavy'] - 1}")

if __name__ == "__main__":
    download_and_slice()