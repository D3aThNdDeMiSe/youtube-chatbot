# YouTube Transcript QA Pipeline with Adaptive CAG/RAG Routing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.51+-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An adaptive question-answering pipeline for YouTube videos that routes each request to the most efficient retrieval strategy based on video length. Short videos use **Cache-Augmented Generation (CAG)** with KV-cache prefill; longer videos use **Retrieval-Augmented Generation (RAG)** with semantic chunking and vector search.

---

## 🎯 Project Overview

This project turns a YouTube URL into a local QA system over the video transcript. Instead of forcing one retrieval approach for every video length, the pipeline automatically selects:

- ✅ **CAG (≤ 10 min):** Prefill the transcript into a KV cache once, then answer follow-up questions without re-encoding the full context
- ✅ **RAG (> 10 min):** Semantic chunking → Chroma embeddings → top-k retrieval at query time
- ✅ **Robust transcript ingestion:** YouTube captions first, faster-whisper fallback when captions are unavailable
- ✅ **Local-first execution:** Runs on consumer GPU hardware with 8-bit Llama inference and CPU offload support
- ✅ **Artifact caching:** Transcripts, KV caches, and vector indexes persist across runs
- ✅ **Notebook + CLI:** Prototype notebook preserved; production path exposed through `main.py`

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Models & Stack](#-models--stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [CLI Reference](#-cli-reference)
- [Pipeline Details](#-pipeline-details)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [Troubleshooting](#-troubleshooting)
- [References](#-references)
- [Contact](#-contact)

---

## 🏗 Architecture

```
                         YouTube URL
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Transcript Ingestion │
                   │  (captions / Whisper) │
                   └──────────┬───────────┘
                              │
                     duration check
                              │
              ┌───────────────┴───────────────┐
              │                               │
     audio_length ≤ 10 min           audio_length > 10 min
              │                               │
              ▼                               ▼
   ┌─────────────────────┐        ┌─────────────────────┐
   │   CAG Pipeline      │        │   RAG Pipeline      │
   │                     │        │                     │
   │ system + transcript │        │ semantic chunker    │
   │  → KV cache prefill │        │  → Qwen3 embeddings │
   │  → cached query gen │        │  → Chroma retrieval │
   └──────────┬──────────┘        └──────────┬──────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
                   ┌──────────────────────┐
                   │ Llama 3.2 3B Instruct │
                   │   (8-bit generation)  │
                   └──────────┬───────────┘
                              ▼
                        Natural-language
                           answer
```

---

## ✨ Key Features

### 1. Adaptive Routing
- **10-minute threshold** switches between CAG and RAG automatically
- Avoids unnecessary vector indexing for short-form content
- Keeps long-form videos tractable via retrieval instead of full-context prefill

### 2. CAG for Short Videos
- Chat-template-aligned KV cache prefill (system prompt + transcript)
- Prefix tokens cached to disk under `KV_Caches/`
- Query-time generation appends only new user tokens to the cached prefix
- Versioned cache invalidation via `CAG_CACHE_VERSION`

### 3. RAG for Long Videos
- Semantic segmentation with `cnmoro/Qwen0.5b-RagSemanticChunker`
- Dense retrieval using `Qwen/Qwen3-Embedding-0.6B`
- Chroma persistence for repeatable querying without re-indexing

### 4. Transcript Resilience
- Primary path: `youtube-transcript-api` manual English captions
- Fallback path: audio download via `yt-dlp` + local `faster-whisper` transcription
- Transcripts cached under `transcripts/{video_id}.txt`

### 5. Local Hardware Awareness
- 8-bit quantized Llama 3.2 3B Instruct
- CPU offload enabled for tight VRAM budgets (~8 GB laptop GPUs)
- Chunking model unloaded after RAG indexing to free GPU memory

---

## 🧠 Models & Stack

| Component | Model / Tool | Role |
|-----------|--------------|------|
| Generation | `meta-llama/Llama-3.2-3B-Instruct` | Answer synthesis (8-bit) |
| Semantic chunking | `cnmoro/Qwen0.5b-RagSemanticChunker` | RAG document segmentation |
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` | Vector indexing + retrieval |
| Transcription fallback | `faster-whisper-large-v3-turbo` | Local ASR when captions missing |
| Vector store | ChromaDB | Persistent chunk retrieval |
| Orchestration | LangChain + Hugging Face Transformers | Pipeline glue |

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU
- Hugging Face account with access to Llama 3.2

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/D3aThNdDeMiSe/youtube-chatbot.git
cd youtube-chatbot
```

2. **Create and activate a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
copy .env.example .env      # Windows
# cp .env.example .env      # Linux/macOS
```
Add your Hugging Face token to `.env`:
```env
HF_TOKEN=your_huggingface_token_here
```

5. **Authenticate with Hugging Face**
```bash
huggingface-cli login
```

6. **Download the Whisper fallback model** (local directory expected at project root)
```bash
# Place faster-whisper-large-v3-turbo/ in the project root
# or update WHISPER_MODEL_PATH in youtube_chatbot/config.py
```

---

## ⚡ Quick Start

### Short video (CAG path)

```bash
python main.py "https://www.youtube.com/watch?v=gJfySWd2HVI" --query "What did Rachel do?"
```

Expected flow:
1. Transcript loaded from cache or fetched
2. KV cache built (first run) or loaded (subsequent runs)
3. Answer generated from cached prefix + user query

### Long video (RAG path)

```bash
python main.py "https://www.youtube.com/watch?v=LONG_VIDEO_ID" --query "Summarize the main argument"
```

Expected flow:
1. Transcript fetched
2. Semantic chunks created and embedded into Chroma
3. Top-k chunks retrieved and passed to Llama for generation

---

## 🖥 CLI Reference

```bash
python main.py URL [--query QUESTION] [--max-new-tokens N]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `URL` | required | YouTube video URL |
| `--query` | `"What did Rachel do?"` | Natural-language question about the video |
| `--max-new-tokens` | `256` | Maximum tokens to generate |

---

## 🔬 Pipeline Details

### CAG prefill contract
The cache is built from the **chat-formatted system message** containing the transcript — not raw transcript tokens. This alignment is critical; mismatched prefill/query formats cause degenerate repetition during generation.

### Cache artifacts
| Path | Purpose |
|------|---------|
| `KV_Caches/{video_id}.pt` | Serialized KV cache |
| `Transcript_Lengths/{video_id}_length.pt` | Prefix token length |
| `Transcript_Lengths/{video_id}_cache_version.pt` | Cache format version |

Bump `CAG_CACHE_VERSION` in `youtube_chatbot/config.py` after changing prefill format.

### RAG indexing
| Path | Purpose |
|------|---------|
| `chroma_db/` | Persistent vector store |
| `transcripts/` | Cached transcript text |

---

## 📁 Project Structure

```
youtube-chatbot/
│
├── main.py                         # CLI entry point
├── setup.ipynb                     # Original research notebook
├── requirements.txt
├── .env.example
├── ROADMAP.md                      # Planned multimodal extensions
│
└── youtube_chatbot/
    ├── config.py                   # Thresholds, model names, paths
    ├── transcript.py               # Caption fetch + Whisper fallback
    ├── cag.py                      # KV cache prefill + cached querying
    ├── rag.py                      # Chunking, Chroma indexing, RAG query
    └── model.py                    # Llama loader (8-bit + CPU offload)
```

---

## 🛣 Roadmap

See [ROADMAP.md](ROADMAP.md) for the full plan. Upcoming work includes:

- [ ] **Multimodal context:** Attach OpenScene / video-frame snippets alongside transcript segments
- [ ] **Multi-granularity transcripts:** Coarse, semi-fine, and fine-grained transcript views
- [ ] **Topic-aware frame selection:** Segment by topic first, then pick representative frames per topic
- [ ] **Interactive UI:** Streamlit or FastAPI front-end for URL + chat input
- [ ] **API deployment:** Wrap pipeline behind a REST endpoint for demo hosting

---

## 🔧 Troubleshooting

### `ValueError: Some modules are dispatched on the CPU or the disk`
Your GPU does not have enough VRAM to hold the full 8-bit model. CPU offload is already enabled in `model.py`. Close other GPU-heavy processes or restart the Python session before loading Llama.

### `NameError: name 'chunks' is not defined`
You are running the RAG indexing path on a video under 10 minutes. Short videos use CAG only — run `main.py` directly instead of manually executing RAG-only steps.

### KV cache outputs repeat tokens (`"to to to..."`)
Delete stale cache files for that `video_id` and rebuild. This usually means an old cache was created before chat-template-aligned prefill was implemented.

### `UnpicklingError` when loading `.pt` cache files
PyTorch 2.6+ defaults to `weights_only=True`. This project loads locally generated caches with `weights_only=False`.

### Whisper fallback fails
Install `ffmpeg` and ensure `faster-whisper-large-v3-turbo/` exists at the configured path.

---

## 📚 References

- **CAG:** [Cache-Augmented Generation overview](https://arxiv.org/abs/2412.15605)
- **Llama 3.2:** [Meta Llama model card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **Semantic chunker:** [cnmoro/Qwen0.5b-RagSemanticChunker](https://huggingface.co/cnmoro/Qwen0.5b-RagSemanticChunker)
- **Embeddings:** [Qwen3 Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- **Faster Whisper:** [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)

---

## 👥 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 📧 Contact

**Immad Shah (D3aThNdDeMiSe)**

- GitHub: [@D3aThNdDeMiSe](https://github.com/D3aThNdDeMiSe)
- LinkedIn: [Immad Shah](https://www.linkedin.com/in/immad-shah-719422294/)
- Email: immadshah18@gmail.com

If this repo helps your own CAG/RAG experiments, a ⭐ on GitHub is appreciated.

---

## 🙏 Acknowledgments

- Prototype notebook iteration and KV-cache debugging during internship R&D
- Hugging Face model hub for Llama, Qwen embedding, and semantic chunking checkpoints
- Related portfolio work: [Federated Learning MLOps Pipeline for E-commerce Customer Churn Prediction](https://github.com/D3aThNdDeMiSe/Federated-Learning-MLOps-Pipeline-for-E-commerce-Customer-Churn-Prediction)
