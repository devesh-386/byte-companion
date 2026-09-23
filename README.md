# Byte: a fully local AI companion

A pixel-art sidekick who lives in the corner of the screen, talks, listens, remembers, and can act on
the laptop. **No API keys and no cloud inference.** Everything runs on this laptop's RTX 5050 (8 GB).
The internet is used only when Byte searches the web for fresh information.

Built as a learning project: every model-shaped piece either was trained here or has a documented
reason for being pretrained.

```
                      ┌────────────── desktop UI (PySide6) ──────────────┐
  mic ─► Whisper ────►│  input  ─►  Agent loop  ─►  bubble + sprite + HUD │──► Piper voice ─► speakers
                      └─────────────────┬────────────────────────────────┘
                                        │ each turn
          ┌──────────────┬──────────────┼───────────────┬──────────────────┐
     EmotionHook     MemoryHook      llama-server       ToolRegistry
     (trained here)  (retriever       Qwen2.5-3B Q4      15 tools: time, maths, files,
     6 reactions     built here,      + persona LoRA     folders, apps, web search/read,
                     bge embeddings)  (trained here)     system info, reminders, memory
```

## Run it

| What | Command |
|---|---|
| Desktop companion | double-click **Byte** on the Desktop, or `start_byte.bat` |
| Terminal chat | `.venv\Scripts\python.exe -m companion` |
| Tests (no GPU needed for most) | `.venv\Scripts\python.exe -m pytest -q` |

Desktop controls: click Byte to show or hide the chat · drag Byte to move it · **●** button to talk (it stops
listening when you pause) · click Byte while it's speaking to stop it · tray icon → speak replies on/off,
new chat, quit. Anything that changes the computer (writing files, opening files, forgetting a memory)
shows Allow / Deny first.

User data lives in `%USERPROFILE%\.companion\` (memory.db, XP/level, window position), separate from code.
Logs: `logs\ui.log`, `logs\llama-server.log`.

## The phases

| Phase | What | Where | Result |
|---|---|---|---|
| 0 | llama.cpp (CUDA 13.4 prebuilt) + Qwen2.5-3B-Instruct Q4_K_M | `llama.cpp-bin/`, `models/` | ~62 tok/s on the GPU |
| 1 | Agent loop, hand-written Qwen tool-call protocol, tool registry | `companion/agent.py`, `protocol.py`, `tools.py` | streams, calls tools, errors go back to the model |
| 2 | **Emotion classifier trained from scratch**: own BPE tokenizer and own Transformer (3.3M params) | `companion/ml/`, `training/emotion/` | test acc **73.7%**, macro-F1 0.685 (majority baseline 39%) |
| 3 | **Long-term memory**: SQLite + bge-small embeddings + own hybrid retriever (rescaled cosine, BM25, recency, MMR) + background fact extraction | `companion/memory/` | recalls facts across chats, stays quiet on unrelated input |
| 4 | **Personality LoRA** (QLoRA on RTX 5050) by context distillation | `training/persona/` | see `artifacts/persona/eval.md` |
| 5 | Desktop companion: procedural pixel sprite, moods, HUD/XP, tools that act on the PC | `companion/ui/`, `companion/plugins/` | |
| 6 | Voice: faster-whisper (ears) + Piper (voice), both offline on CPU | `companion/voice.py` | ~0.8 s to transcribe, ~0.8 s to speak a sentence |

## Lessons that came out of building it

- **Domain shift** (phase 2): trained on Reddit and tweets, the classifier called "I GOT THE INTERNSHIP!!!"
  neutral. 133 hand-written companion-style sentences fixed it. Questions had been labelled "curiosity",
  mapped to surprise; for a companion, a question is neutral.
- **Calibrate before you threshold** (phase 3): bge scores *unrelated* sentences at 0.35–0.52, so raw
  cosine is misleading. The retriever rescales it, and saturates BM25 so one shared word ("called")
  can't look like a perfect match.
- **Few-shot beats instructions for small models** (phase 3): fact extraction found nothing until it
  was shown five worked examples.
- **Tool output wording matters** (phase 1): "228 items" became "228 folders" until the tool said
  "16 folders and 212 files".
- **Memory on an 8 GB GPU** (phase 4): the first QLoRA attempt used 14 GB. The model was in eval mode, so
  gradient checkpointing silently did nothing. With `model.train()` and logits only where the loss is
  graded, peak dropped to 3.1 GB.

## Rebuilding from scratch

Large files are not in git. To rebuild:

```
python -m venv --system-site-packages .venv      # reuses the system CUDA PyTorch
.venv\Scripts\python.exe -m pip install transformers peft accelerate bitsandbytes safetensors huggingface_hub pyarrow gguf sentencepiece PySide6 faster-whisper piper-tts
# llama.cpp b11118 win-cuda-13.4 x64 + cudart zip -> llama.cpp-bin/runtime/  (gh release download)
# models/qwen2.5-3b-instruct-q4_k_m.gguf, models/Qwen2.5-3B-Instruct (HF), models/bge-small-en-v1.5
.venv\Scripts\python.exe -m piper.download_voices en_US-lessac-medium --download-dir models/piper
.venv\Scripts\python.exe training/emotion/prepare_data.py && .venv\Scripts\python.exe training/emotion/train.py
.venv\Scripts\python.exe training/persona/generate_data.py && .venv\Scripts\python.exe training/persona/train_lora.py
.venv\Scripts\python.exe training/persona/export_lora.py && .venv\Scripts\python.exe training/persona/eval_persona.py
```

## Known limits

- The 3B brain is small: good at chat and tool use, weak at long reasoning. The architecture doesn't
  care which GGUF model is loaded, so a bigger one can be swapped in.
- No screen vision yet (a local vision model would be the next phase).
- Per-turn context (mood, memories) sits in the system prompt, which stops llama-server reusing its
  cached prompt across turns. Moving it into the user turn needs the persona adapter retrained to match.
- Reminders only fire while Byte is running.
