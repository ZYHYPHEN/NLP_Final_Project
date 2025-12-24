import os
import numpy as np
import librosa
import torch
import whisper
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from jiwer import wer, cer
import shutil # Added to check for ffmpeg

# =================CONFIGURATION=================
# Directories containing your data
# Adjust these paths if your actual folder names are different
DATASET_ROOT = r"F:\CUHKSZ\CSC5051_NLP\Project\dataset"
GENERATED_DIR = os.path.join(DATASET_ROOT, "result")   # Generated audio (1.wav, 2.wav...)
REFERENCE_DIR = DATASET_ROOT                           # Original audio (1.wav, 2.wav...) or os.path.join(DATASET_ROOT, "reference")
LABEL_FILE = os.path.join(DATASET_ROOT, "label.txt")

# Set to True to print detailed metrics for every file
VERBOSE = True
# ===============================================

def check_ffmpeg():
    """Checks if ffmpeg is installed and accessible."""
    if not shutil.which("ffmpeg"):
        print("[Error] ffmpeg not found in system PATH. Whisper requires ffmpeg.")
        print("Please install ffmpeg and add it to your PATH environment variable.")
        return False
    return True

def calculate_mcd(ref_wav, gen_wav, sr=16000, n_mfcc=13):
    """Calculates Mel-Cepstral Distortion (MCD) using DTW."""
    try:
        y_ref, _ = librosa.load(ref_wav, sr=sr)
        y_gen, _ = librosa.load(gen_wav, sr=sr)

        # Extract MFCCs
        mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=n_mfcc).T
        mfcc_gen = librosa.feature.mfcc(y=y_gen, sr=sr, n_mfcc=n_mfcc).T

        # Align using DTW
        distance, path = fastdtw(mfcc_ref, mfcc_gen, dist=euclidean)
        
        # MCD calculation (approximate constant 6.14)
        mcd = 6.14 * (distance / len(path))
        return mcd
    except Exception as e:
        print(f"[Error] MCD calculation failed for {os.path.basename(gen_wav)}: {e}")
        return None

def calculate_f0_rmse(ref_wav, gen_wav, sr=16000):
    """Calculates RMSE of Fundamental Frequency (Pitch)."""
    try:
        y_ref, _ = librosa.load(ref_wav, sr=sr)
        y_gen, _ = librosa.load(gen_wav, sr=sr)

        # Extract Pitch using pyin
        f0_ref, _, _ = librosa.pyin(y_ref, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_gen, _, _ = librosa.pyin(y_gen, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

        # Remove unvoiced regions (NaNs)
        f0_ref = f0_ref[~np.isnan(f0_ref)]
        f0_gen = f0_gen[~np.isnan(f0_gen)]

        if len(f0_ref) == 0 or len(f0_gen) == 0:
            return 0.0

        # Simple difference of means for non-aligned audio
        return abs(np.mean(f0_ref) - np.mean(f0_gen))
    except Exception as e:
        print(f"[Error] F0 calculation failed for {os.path.basename(gen_wav)}: {e}")
        return None

def calculate_asr_metrics(audio_path, target_text, model):
    """Calculates CER (Character Error Rate) using Whisper."""
    try:
        # Transcribe
        result = model.transcribe(audio_path, language="zh") # Force Chinese
        hypothesis = result["text"]
        
        # For Chinese, CER is usually the standard metric instead of WER
        # WER might split by space which doesn't exist in Chinese sentences
        error_rate = cer(target_text, hypothesis)
        return error_rate, hypothesis
    except Exception as e:
        print(f"[Error] ASR failed for {os.path.basename(audio_path)}: {e}")
        return None, ""

def load_labels(label_path):
    """Reads label.txt and returns a dictionary {id: text}."""
    labels = {}
    if not os.path.exists(label_path):
        print(f"[Error] Label file not found at: {label_path}")
        return labels
        
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1) # Split by first whitespace/tab
            if len(parts) == 2:
                file_id, text = parts
                labels[file_id] = text
            elif len(parts) == 1:
                 # Handle case where split might fail or formatting is different
                 print(f"[Warning] Skipping malformed line: {line.strip()}")
    return labels

def main():
    print(f"Dataset Root: {DATASET_ROOT}")
    
    # Check for ffmpeg first
    if not check_ffmpeg():
        return

    print(f"Loading Labels from: {LABEL_FILE} ...")
    labels = load_labels(LABEL_FILE)
    if not labels:
        print("No labels found. Exiting.")
        return

    print("Loading Whisper Model (base)...")
    try:
        asr_model = whisper.load_model("base")
    except Exception as e:
        print(f"[Fatal Error] Failed to load Whisper model: {e}")
        print("Ensure you have openai-whisper installed correctly and internet access to download the model.")
        return

    # Metrics storage
    metrics = {
        "CER": [],
        "MCD": [],
        "F0_RMSE": []
    }

    print("\n" + "="*60)
    print(f"{'ID':<5} | {'CER':<8} | {'MCD':<8} | {'F0 Diff':<8} | {'Text (Ref vs Gen)'}")
    print("="*60)

    for file_id, target_text in labels.items():
        # Construct file paths
        # Assuming audio files are named "1.wav", "2.wav" etc.
        gen_path = os.path.join(GENERATED_DIR, f"{file_id}.wav")
        
        # Check reference location. You mentioned "1.wav" is in dataset root in your `ls` output
        # If they are in a 'reference' subfolder, change this line.
        ref_path = os.path.join(REFERENCE_DIR, f"{file_id}.wav") 
        if not os.path.exists(ref_path):
             # Try 'reference' folder fallback if not in root
             ref_path_alt = os.path.join(DATASET_ROOT, "reference", f"{file_id}.wav")
             if os.path.exists(ref_path_alt):
                 ref_path = ref_path_alt

        # 1. Evaluate Intelligibility (CER)
        cer_score = None
        hypothesis = ""
        if os.path.exists(gen_path):
            cer_score, hypothesis = calculate_asr_metrics(gen_path, target_text, asr_model)
            if cer_score is not None:
                metrics["CER"].append(cer_score)
        else:
            print(f"[Warning] Generated file not found: {gen_path}")
            continue

        # 2. Evaluate Similarity (MCD & F0)
        mcd_val = None
        f0_val = None
        
        if os.path.exists(ref_path):
            mcd_val = calculate_mcd(ref_path, gen_path)
            f0_val = calculate_f0_rmse(ref_path, gen_path)
            
            if mcd_val is not None: metrics["MCD"].append(mcd_val)
            if f0_val is not None: metrics["F0_RMSE"].append(f0_val)
        else:
            if VERBOSE: print(f"[Info] Ref file not found for ID {file_id}, skipping similarity metrics.")

        # Print row
        # Fixed formatting for potential None/String values
        cer_str = f"{cer_score:.4f}" if cer_score is not None else "-"
        mcd_str = f"{mcd_val:.4f}" if mcd_val is not None else "-"
        f0_str = f"{f0_val:.4f}" if f0_val is not None else "-"

        print(f"{file_id:<5} | {cer_str:<8} | {mcd_str:<8} | {f0_str:<8} | Ref: {target_text}")
        print(f"{'':<5} | {'':<8} | {'':<8} | {'':<8} | Gen: {hypothesis}")
        print("-" * 60)

    # --- Summary ---
    print("\n" + "="*30)
    print("FINAL QUANTITATIVE RESULTS")
    print("="*30)
    
    if metrics["CER"]:
        print(f"Average CER (Intelligibility): {np.mean(metrics['CER']):.4f} (Lower is better)")
    
    if metrics["MCD"]:
        print(f"Average MCD (Timbre Sim.):   {np.mean(metrics['MCD']):.4f} (Lower is better)")
    
    if metrics["F0_RMSE"]:
        print(f"Average F0 Diff (Pitch Sim.):  {np.mean(metrics['F0_RMSE']):.4f} Hz (Lower is better)")

if __name__ == "__main__":
    main()