# YouTube Chatbot

Ask questions about YouTube videos using their transcripts. Automatically picks the best retrieval strategy based on video length:

- **≤ 10 minutes** → **CAG** (Cache-Augmented Generation): prefill transcript into a KV cache, then answer queries without re-processing the full transcript
- **> 10 minutes** → **RAG**: semantic chunking → Chroma embeddings → retrieve top chunks at query time

## Setup

1. Clone the repo and create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and add your Hugging Face token (required for Llama 3.2):

```bash
copy .env.example .env
```

3. Download the Whisper fallback model (used when YouTube captions aren't available):

```bash
# Example: clone or download faster-whisper-large-v3-turbo into the project root
```

4. Log in to Hugging Face if you haven't already:

```bash
huggingface-cli login
```

## Usage

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --query "What did Rachel do?"
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--query` | `"What did Rachel do?"` | Question to ask about the video |
| `--max-new-tokens` | `256` | Max tokens to generate |

## Project structure

```
main.py                  # CLI entry point
youtube_chatbot/
  config.py              # Constants and paths
  transcript.py          # Transcript fetching (YouTube API + Whisper fallback)
  rag.py                 # Semantic chunking, Chroma indexing, RAG query
  cag.py                 # KV cache prefill and CAG query
  model.py               # Llama 3.2 3B Instruct loader
setup.ipynb              # Original prototype notebook
```

## Models

| Purpose | Model |
|---------|-------|
| Query / generation | `meta-llama/Llama-3.2-3B-Instruct` (8-bit) |
| Semantic chunking (RAG) | `cnmoro/Qwen0.5b-RagSemanticChunker` |
| Embeddings (RAG) | `Qwen/Qwen3-Embedding-0.6B` |
| Transcription fallback | `faster-whisper-large-v3-turbo` (local) |

## Notes

- Cached KV states are stored under `KV_Caches/` and `Transcript_Lengths/`.
- Chroma vectors persist under `chroma_db/`.
- Transcripts are cached under `transcripts/`.
- Bump `CAG_CACHE_VERSION` in `youtube_chatbot/config.py` if the prefill format changes.
