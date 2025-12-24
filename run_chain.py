import os
import sys
import gc
import torch
import numpy as np
import soundfile as sf
import importlib
from time import time as ttime

# Set parallelism to false to prevent segfaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add current directory to path
now_dir = os.getcwd()
sys.path.append(now_dir)
# CRITICAL FIX: Add GPT_SoVITS subdirectory to path so inner modules can find 'utils'
sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))

# ==========================================
# ### CONFIGURATION ###
# Edit these paths to match your server environment
# ==========================================

# 1. LLAMA 3 CONFIG
LLAMA_MODEL_PATH = './Llama3'
LORA_PATH = './checkpoint495' 
SYSTEM_PROMPT = "假设你是西游记中的孙悟空，请以孙悟空的口吻回答我。"
DEFAULT_USER_PROMPT = "悟空，你为何在五行山下啊？" #大王何为烦恼，为何流泪？ 悟空，你为何在五行山下啊？
USER_PROMPT = ""

# 2. GPT-SOVITS MODEL CONFIG
# Path to the SoVITS weights (*.pth)
SOVITS_PATH = "./voice_model/sunwukong_sovits.pth" 
# Path to the GPT weights (*.ckpt)
GPT_PATH = "./voice_model/sunwukong_gpt.ckpt" 

# 3. GPT-SOVITS REFERENCE AUDIO CONFIG
# The voice you want to clone
REF_AUDIO_PATH = "./ref_audio/wukong_ref.mp3" 
REF_TEXT = "嘿嘿！我乃五百年前大闹天宫的齐天大圣，孙悟空！"
REF_LANGUAGE = "中文" # options: 中文, 英文, 日文

# 4. SHARED CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_HALF = True  # Use fp16 for faster inference (set False if using CPU)
OUTPUT_WAV_PATH = "./output/output_response.wav"

# Pretrained models for GPT-SoVITS (usually in GPT_SoVITS/pretrained_models)
CNHUBERT_PATH = "./GPT_SoVITS/pretrained_models/chinese-hubert-base"
BERT_PATH = "./GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"

# ==========================================
# AUTO-DETECT MODEL VERSION & CLEANUP
# ==========================================
# We must detect V1 vs V2 BEFORE importing GPT_SoVITS modules
model_version = "v2" # Default
try:
    # print(f"[Init] Inspecting model version from: {SOVITS_PATH}")
    # weights_only=False required to load legacy checkpoints with HParams
    checkpoint = torch.load(SOVITS_PATH, map_location="cpu", weights_only=False)
    
    if "weight" in checkpoint:
        state_dict = checkpoint["weight"]
        if "enc_p.text_embedding.weight" in state_dict:
            emb_shape = state_dict["enc_p.text_embedding.weight"].shape
            vocab_size = emb_shape[0]
            
            
            if vocab_size == 322:
                os.environ["version"] = "v1"
                model_version = "v1"
            elif vocab_size == 732:
                os.environ["version"] = "v2"
                model_version = "v2"
            else:
                print(f"[Init] Unknown vocabulary size {vocab_size}. Defaulting to v2.")
                os.environ["version"] = "v2"
        else:
            print("[Init] Could not find text embeddings. Defaulting to v2.")
            os.environ["version"] = "v2"
    else:
        print("[Init] 'weight' key not found in checkpoint. Defaulting to v2.")
        os.environ["version"] = "v2"
        
    del checkpoint
    gc.collect()
except Exception as e:
    print(f"[Init] Error detecting version: {e}. Defaulting to v2.")
    os.environ["version"] = "v2"

# CRITICAL: Aggressively unload any GPT_SoVITS modules that might have been auto-loaded
# to ensure they reload with the new os.environ['version'] settings.
modules_to_unload = []
for m in sys.modules:
    if m.startswith("GPT_SoVITS") or m.startswith("module") or m.startswith("AR") or m.startswith("text") or m.startswith("feature_extractor"):
        modules_to_unload.append(m)
for m in modules_to_unload:
    del sys.modules[m]


# ==========================================
# IMPORTS (After Version Set & Cleanup)
# ==========================================

# CRITICAL: Monkeypatch text.symbols if needed BEFORE importing models
# The models.py file imports 'from text import symbols'. 
# If that file contains V2 symbols (732) but we want V1 (322), we must patch it.
try:
    # Ensure text.symbols is loaded as 'text.symbols' matching models.py import
    import text.symbols
    current_vocab_size = len(text.symbols.symbols)
    #print(f"[Init] Loaded text.symbols. Vocab Size: {current_vocab_size}")
    
    if model_version == "v1" and current_vocab_size != 322:
        print("[Warning] Symbols mismatch! Forcing V1 symbols list in 'text.symbols'...")
        # Force the symbol list to be the V1 length (322)
        text.symbols.symbols = text.symbols.symbols[:322]
except ImportError:
    print("[Init] Could not import 'text.symbols' directly. Skipping patch.")

from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.module.models import SynthesizerTrn
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.module.mel_processing import spectrogram_torch
from transformers import AutoModelForMaskedLM, AutoTokenizer as BertTokenizer

# ==========================================
# PART 1: LLAMA 3 (Isolated)
# ==========================================

def run_llama_stage():
    print("-" * 30)
    print("[Stage 1] Starting Llama 3 Generation...")
    print("-" * 30)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    response_text = ""
    
    try:
        print(f"[LLM] Loading tokenizer from {LLAMA_MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH, trust_remote_code=True)
        
        print(f"[LLM] Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_PATH, 
            device_map="auto",
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        ).eval()
        
        if LORA_PATH and os.path.exists(LORA_PATH):
            print(f"[LLM] Loading LoRA adapters...")
            model = PeftModel.from_pretrained(model, model_id=LORA_PATH)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ]
        
        input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([input_ids], return_tensors="pt").to(DEVICE)
        
        print(f"[LLM] Generating...")
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"[LLM] Generated Text: {response_text}")

        # CLEANUP - CRITICAL FOR AVOIDING SEGFAULT
        del model
        del tokenizer
        del model_inputs
        del generated_ids
        
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return None

    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("[Stage 1] Complete. Memory Cleared.")
    return response_text

# ==========================================
# PART 2: GPT-SOVITS (Isolated)
# ==========================================

def run_tts_stage(text):
    print("-" * 30)
    print("[Stage 2] Starting GPT-SoVITS Generation...")
    print("-" * 30)
    
    # Delayed imports to avoid interference with Llama
    import librosa

    # Define helper function locally since my_utils is missing
    def load_audio(file, sr):
        try:
            file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            audio, _ = librosa.load(file, sr=sr, mono=True)
            return audio
        except Exception as e:
            print(f"Error loading audio: {e}")
            return np.zeros(0)

    try:
        # 1. Init Models
        print("[TTS] Initializing CNHubert...")
        cnhubert.cnhubert_base_path = CNHUBERT_PATH
        ssl_model = cnhubert.get_model()
        if IS_HALF:
            ssl_model = ssl_model.half().to(DEVICE)
        else:
            ssl_model = ssl_model.to(DEVICE)

        print("[TTS] Initializing BERT...")
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        bert_model = AutoModelForMaskedLM.from_pretrained(BERT_PATH)
        if IS_HALF:
            bert_model = bert_model.half().to(DEVICE)
        else:
            bert_model = bert_model.to(DEVICE)

        print("[TTS] Initializing SoVITS...")
        # Note: weights_only=False enabled to handle custom HParams class
        dict_s2 = torch.load(SOVITS_PATH, map_location="cpu", weights_only=False)
        hps = dict_s2["config"]
        
        class DictToAttrRecursive(dict):
            def __init__(self, input_dict):
                super().__init__(input_dict)
                for key, value in input_dict.items():
                    if isinstance(value, dict):
                        value = DictToAttrRecursive(value)
                    self[key] = value
                    setattr(self, key, value)
        
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"

        # Force V1 configuration if detected
        if model_version == "v1":
             # SynthesizerTrn uses 'version' arg to select symbol set.
             # We inject it here to ensure it uses the logic in models.py
             hps.model['version'] = "v1"
        
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        if IS_HALF:
            vq_model = vq_model.half().to(DEVICE)
        else:
            vq_model = vq_model.to(DEVICE)
        vq_model.eval()
        vq_model.load_state_dict(dict_s2["weight"], strict=False)

        print("[TTS] Initializing GPT...")
        dict_s1 = torch.load(GPT_PATH, map_location="cpu", weights_only=False)
        config = dict_s1["config"]
        t2s_model = Text2SemanticLightningModule(config, "ojbk", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        if IS_HALF:
            t2s_model = t2s_model.half()
        t2s_model = t2s_model.to(DEVICE)
        t2s_model.eval()

        # Helper functions
        def get_bert_feature(text, word2ph):
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt")
                for i in inputs:
                    inputs[i] = inputs[i].to(DEVICE)
                res = bert_model(**inputs, output_hidden_states=True)
                res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
            assert len(word2ph) == len(text)
            phone_level_feature = []
            for i in range(len(word2ph)):
                repeat_feature = res[i].repeat(word2ph[i], 1)
                phone_level_feature.append(repeat_feature)
            phone_level_feature = torch.cat(phone_level_feature, dim=0)
            return phone_level_feature.T

        def get_spepc(hps, filename):
            audio = load_audio(filename, int(hps.data.sampling_rate))
            audio = torch.FloatTensor(audio)
            audio_norm = audio.unsqueeze(0)
            spec = spectrogram_torch(
                audio_norm, 
                hps.data.filter_length, 
                hps.data.sampling_rate, 
                hps.data.hop_length,
                hps.data.win_length, 
                center=False
            )
            return spec

        # Inference Logic
        dict_language = {
            "中文": "zh", "英文": "en", "日文": "ja",
            "ZH": "zh", "EN": "en", "JA": "ja",
            "zh": "zh", "en": "en", "ja": "ja"
        }
        
        text_language = "zh" # Force Chinese for now based on Llama output
        prompt_language = dict_language[REF_LANGUAGE]
        
        # 1. Reference Audio
        zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if IS_HALF else np.float32)
        with torch.no_grad():
            wav16k, sr = librosa.load(REF_AUDIO_PATH, sr=16000)
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if IS_HALF:
                wav16k = wav16k.half().to(DEVICE)
                zero_wav_torch = zero_wav_torch.half().to(DEVICE)
            else:
                wav16k = wav16k.to(DEVICE)
                zero_wav_torch = zero_wav_torch.to(DEVICE)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]

        # 2. Text
        phones1, word2ph1, norm_text1 = clean_text(REF_TEXT, prompt_language)
        phones1 = cleaned_text_to_sequence(phones1)
        phones2, word2ph2, norm_text2 = clean_text(text, text_language)
        phones2 = cleaned_text_to_sequence(phones2)

        # 3. BERT
        if prompt_language == "zh":
            bert1 = get_bert_feature(norm_text1, word2ph1).to(DEVICE)
        else:
            bert1 = torch.zeros((1024, len(phones1)), dtype=torch.float16 if IS_HALF else torch.float32).to(DEVICE)
        
        if text_language == "zh":
            bert2 = get_bert_feature(norm_text2, word2ph2).to(DEVICE)
        else:
            bert2 = torch.zeros((1024, len(phones2))).to(bert1)
            
        bert = torch.cat([bert1, bert2], 1)
        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(DEVICE).unsqueeze(0)
        bert = bert.to(DEVICE).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(DEVICE)
        prompt = prompt_semantic.unsqueeze(0).to(DEVICE)

        # 4. GPT
        with torch.no_grad():
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                top_k=config['inference']['top_k'],
                early_stop_num=50 * config['data']['max_sec']
            )

        # 5. SoVITS
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
        refer = get_spepc(hps, REF_AUDIO_PATH)
        if IS_HALF:
            refer = refer.half().to(DEVICE)
        else:
            refer = refer.to(DEVICE)

        audio = vq_model.decode(
            pred_semantic, 
            torch.LongTensor(phones2).to(DEVICE).unsqueeze(0),
            refer
        ).detach().cpu().numpy()[0, 0]

        return hps.data.sampling_rate, (audio * 32768).astype(np.int16)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[TTS] Error: {e}")
        return None, None

# ==========================================
# MAIN
# ==========================================

def main():
    # print("="*40)
    # print("ENVIRONMENT DIAGNOSTICS")
    # print("="*40)
    # print(f"Python: {sys.version.split()[0]}")
    # print(f"PyTorch Version: {torch.__version__}")
    # print(f"CUDA Available:  {torch.cuda.is_available()}")
    # if torch.cuda.is_available():
    #     print(f"CUDA Version:    {torch.version.cuda}")
    #     print(f"Device Count:    {torch.cuda.device_count()}")
    #     print(f"Current Device:  {torch.cuda.get_device_name(0)}")
    # else:
    #     print("WARNING: Running on CPU. This will be very slow and might be the result of a bad install.")
    # print("="*40 + "\n")
    # 1. Generate Text
    print("="*40 + "\n")
    print(f"Given Reference Text: {REF_TEXT}")
    print("="*40 + "\n")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    while True:
        USER_PROMPT = input("\nUser: ").strip()
        if not USER_PROMPT:
            print(f"Using default Prompt Text: {de_USER_PROMPT}")
            USER_PROMPT = DEFAULT_USER_PROMPT
        print(f"Given Prompt Text: {USER_PROMPT}")
        generated_text = run_llama_stage()
        
        if not generated_text:
            print("Failed to generate text.")
            return
    
        # 2. Generate Audio
        sr, audio_data = run_tts_stage(generated_text)
        
        if audio_data is not None:
            os.makedirs(os.path.dirname(OUTPUT_WAV_PATH), exist_ok=True)
            sf.write(OUTPUT_WAV_PATH, audio_data, sr)
            print(f"\n[Success] Audio saved to {OUTPUT_WAV_PATH}")
        else:
            print("\n[Failed] Audio generation failed.")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()