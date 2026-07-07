from pathlib import Path

CAG_SYSTEM_PROMPT = "Answer questions based on the provided video transcript."
CAG_CACHE_VERSION = 1

RAG_THRESHOLD_SECONDS = 60 * 10

QUERY_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
CHUNKING_MODEL = "cnmoro/Qwen0.5b-RagSemanticChunker"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
WHISPER_MODEL_PATH = "./faster-whisper-large-v3-turbo"

TRANSCRIPTS_DIR = Path("./transcripts")
DOWNLOADS_DIR = Path("./downloads")
KV_CACHES_DIR = Path("./KV_Caches")
TRANSCRIPT_LENGTHS_DIR = Path("./Transcript_Lengths")
CHROMA_DIR = Path("./chroma_db")

CHROMA_COLLECTION = "youtube-transcripts"
