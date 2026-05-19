# Setup Guide

## 1. Python Environment

```bash
uv sync
```

---

## 2. ffmpeg

Required by pydub for audio processing. Install and place at `C:/ffmpeg/`.

Download: https://ffmpeg.org/download.html

Expected paths:
- `C:/ffmpeg/bin/ffmpeg.exe`
- `C:/ffmpeg/bin/ffprobe.exe`

---

## 3. Clone Qwen3-TTS Finetuning Scripts

`02_data_prep.ipynb` does this automatically on first run.  
To do it manually:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/QwenLM/Qwen3-TTS finetune/scripts
git -C finetune/scripts sparse-checkout set finetuning
```

---

## 4. Patch finetune/scripts/finetuning/sft_12hz.py

`dataset.py` and `prepare_data.py` are used as-is. Only `sft_12hz.py` needs changes.

**Add after existing imports:**
```python
from huggingface_hub import snapshot_download
```

**Add inside `train()`, after `MODEL_PATH = args.init_model_path`:**
```python
if not os.path.isdir(MODEL_PATH):
    MODEL_PATH = snapshot_download(MODEL_PATH)
```

**Change `from_pretrained` arguments:**
```python
# from
torch_dtype=torch.bfloat16,
attn_implementation="flash_attention_2",

# to
dtype=torch.bfloat16,
attn_implementation="sdpa",
```

---

## 5. Reference Audio

Record two short clips (~10 seconds each) and place in `audio/`:

| File | Language |
|---|---|
| `audio/laxmikant_en.wav` | English |
| `audio/laxmikant_hi.wav` | Hindi |

Transcripts are already in `audio/laxmikant_en_ref.txt` and `audio/laxmikant_hi_ref.txt`.

---

## 6. Training Data

Record the full script at `script/finetune_script.txt` and place the WAV in `finetune/raw_recordings/`.  
Then run `02_data_prep.ipynb` → `03_finetune.ipynb`.

---

## Notes

- Checkpoints saved to `finetune/checkpoints/checkpoint-epoch-N/`
- Each checkpoint drops `speaker_encoder.*` weights — this is expected
- To resume training, re-inject speaker encoder weights from base model first (see `03_finetune.ipynb` Section 2)
