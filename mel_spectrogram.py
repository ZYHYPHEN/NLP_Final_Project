import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
# Update these paths to your actual files
DATASET_ROOT = r"F:\CUHKSZ\CSC5051_NLP\Project\dataset"
GENERATED_DIR = os.path.join(DATASET_ROOT, "result")
REFERENCE_DIR = DATASET_ROOT # Or "reference" subfolder

# Define the IDs for your case study
BEST_CASE_ID = "4"  # ID 4: "Qi Tian Da Sheng..."
WORST_CASE_ID = "2" # ID 2: "An Lao Sun..."

OUTPUT_FILENAME = "cases_placeholder.png" # Will be saved in current dir
# =================================================

def compute_mel_spectrogram(wav_path, sr=16000, n_mels=80, fmax=8000):
    """Loads audio and computes Mel-spectrogram."""
    try:
        y, _ = librosa.load(wav_path, sr=sr)
        # Standard Mel-spec settings for TTS analysis
        S = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels, 
            fmax=fmax, 
            n_fft=1024, 
            hop_length=256
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

def plot_comparison():
    # Setup paths
    gen_best = os.path.join(GENERATED_DIR, f"{BEST_CASE_ID}.wav")
    ref_best = os.path.join(REFERENCE_DIR, f"{BEST_CASE_ID}.wav")
    
    gen_worst = os.path.join(GENERATED_DIR, f"{WORST_CASE_ID}.wav")
    ref_worst = os.path.join(REFERENCE_DIR, f"{WORST_CASE_ID}.wav")

    # Check existence
    for p in [gen_best, ref_best, gen_worst, ref_worst]:
        if not os.path.exists(p):
            # Try alternate reference path if main fails
            alt_p = p.replace(DATASET_ROOT, os.path.join(DATASET_ROOT, "reference"))
            if not os.path.exists(alt_p) and "result" not in p:
                print(f"[Error] File not found: {p}")
                return

    # Compute Spectrograms
    spec_ref_best = compute_mel_spectrogram(ref_best)
    spec_gen_best = compute_mel_spectrogram(gen_best)
    spec_ref_worst = compute_mel_spectrogram(ref_worst)
    spec_gen_worst = compute_mel_spectrogram(gen_worst)

    if any(x is None for x in [spec_ref_best, spec_gen_best, spec_ref_worst, spec_gen_worst]):
        return

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    
    # Common kwargs for display
    disp_kw = {'x_axis': 'time', 'y_axis': 'mel', 'fmax': 8000, 'cmap': 'magma'}

    # Row 1: Best Case
    librosa.display.specshow(spec_ref_best, ax=axes[0, 0], **disp_kw)
    axes[0, 0].set_title(f"Reference (Best Case ID {BEST_CASE_ID})")
    
    librosa.display.specshow(spec_gen_best, ax=axes[0, 1], **disp_kw)
    axes[0, 1].set_title(f"Generated (Best Case ID {BEST_CASE_ID}) - Low MCD/F0 Diff")

    # Row 2: Worst Case
    librosa.display.specshow(spec_ref_worst, ax=axes[1, 0], **disp_kw)
    axes[1, 0].set_title(f"Reference (Worst Case ID {WORST_CASE_ID})")
    
    librosa.display.specshow(spec_gen_worst, ax=axes[1, 1], **disp_kw)
    axes[1, 1].set_title(f"Generated (Worst Case ID {WORST_CASE_ID}) - High MCD/F0 Diff")

    # Save
    plt.savefig(OUTPUT_FILENAME, dpi=300)
    print(f"[Success] Comparison plot saved to {os.path.abspath(OUTPUT_FILENAME)}")
    plt.show()

if __name__ == "__main__":
    plot_comparison()