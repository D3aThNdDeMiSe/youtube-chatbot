# Roadmap — YouTube Transcript QA Pipeline

Personal development plan for evolving this project from transcript-only QA into a multimodal video understanding system.

---

## ✅ Completed

- [x] YouTube transcript ingestion with caption + Whisper fallback
- [x] Adaptive routing: CAG for videos ≤ 10 min, RAG for longer content
- [x] Chat-template-aligned KV cache prefill (fixes repetition bug)
- [x] Semantic chunking + Chroma indexing for long-form RAG
- [x] Notebook prototype ported to modular Python CLI
- [x] Local 8-bit Llama inference with CPU offload for constrained VRAM

---

## 🚧 In Progress / Next Up

### 1. Multimodal frame context (OpenScene)
- Extract representative video frames alongside transcript segments
- Attach frame descriptions or visual embeddings as additional retrieval context
- Goal: answer questions that depend on on-screen content, not just spoken words

### 2. Multi-granularity transcript views
- Generate three transcript representations:
  - **Coarse:** high-level section summaries
  - **Semi-fine:** paragraph-level chunks
  - **Fine-grained:** sentence-level detail
- Feed all three into a preprocessor that groups content by topic before retrieval

### 3. Topic-aware frame selection
- Replace fixed-duration chunking with topic segmentation
- Select frames per topic rather than at hard-coded time intervals
- Align visual context windows with semantic topic boundaries

---

## 🔮 Future Enhancements

- [ ] FastAPI service wrapping `main.py` for hosted demos
- [ ] Streamlit chat UI for URL input + multi-turn questioning
- [ ] Conversation memory across follow-up queries in CAG mode
- [ ] Cache warm-up endpoint for preloading popular videos
- [ ] Dockerized deployment with optional GPU profile
- [ ] Evaluation harness with timestamp-grounded QA pairs
- [ ] Support for playlist-level indexing and cross-video search

---

## 💡 Research Questions

1. When does CAG outperform RAG for medium-length videos (8–15 min) once KV cache size grows?
2. Can multi-granularity transcripts reduce hallucination on long educational content?
3. How much does frame context improve answer accuracy vs transcript-only baselines?

---

Maintained by [Immad Shah](https://github.com/D3aThNdDeMiSe).
