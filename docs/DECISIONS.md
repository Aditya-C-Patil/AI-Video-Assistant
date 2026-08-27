# 🏛️ Architecture & Engineering Decision Records (ADRs)

This document details the architectural choices, engineering trade-offs, and design rationale behind the **NEXUS AI // Video/Audio Meeting Intelligence & RAG System**.

---

## 1. Speech-to-Text Pipeline: Cloud Groq Whisper vs. Local Engine

| Decision Aspect      | Selection                             | Alternatives Considered                            |
| -------------------- | ------------------------------------- | -------------------------------------------------- |
| **Model & Provider** | `whisper-large-v3` via Groq Cloud API | Local `openai-whisper`, Deepgram, AssemblyAI       |
| **Audio Format**     | 16kHz Mono 16-bit PCM WAV             | MP3, raw AAC, FLAC                                 |
| **Chunking Logic**   | 10-Minute Slices with `pydub`         | Single-shot streaming, sliding window with overlap |

### Rationale & Trade-offs

* **Inference Speed & Resource Utilization:** Running `whisper-large-v3` locally requires high-VRAM GPUs (8GB+ dedicated VRAM) and introduces thermal/latency bottlenecks on client machines. Groq's LPU hardware delivers near-instant transcription (~200x real-time speed) without client hardware constraints.
* **Payload Boundary Management:** Upstream transcription endpoints enforce hard file size limits (25MB). Pre-processing all inputs into 16kHz mono audio and slicing into 10-minute segments prevents connection timeouts and out-of-memory errors on hour-long media.
* **Storage Footprint:** Raw video/audio files are discarded after conversion into temporary chunk directories, followed by automated cleanup via `utils/audio_cleanup.py` to prevent disk bloat.

---

## 2. Multi-Tenant Vector Isolation: Partitioned ChromaDB Collections

| Decision Aspect       | Selection                                  | Alternatives Considered                                          |
| --------------------- | ------------------------------------------ | ---------------------------------------------------------------- |
| **Data Partitioning** | Dynamic `collection_name="{meeting_id}"`   | Single shared collection with metadata `where={"meeting_id": x}` |
| **Embedding Model**   | `all-MiniLM-L6-v2` (Sentence-Transformers) | `text-embedding-3-small`, `bge-large-en`                         |

### Rationale & Trade-offs

* **Zero Context Bleeding:** In enterprise meeting intelligence, retrieval queries must never leak context from one meeting into another. While metadata filtering (`where={"meeting_id": id}`) can filter candidates post-indexing, dynamic per-meeting ChromaDB collections physically separate indices at the vector space level, providing zero risk of cross-meeting leakage and faster query times over smaller indexes.
* **Local Embedding Efficiency:** `all-MiniLM-L6-v2` runs locally on CPU with minimal latency (~15ms per chunk), 384-dimensional embeddings (compact index footprint), and zero API cost.

---

## 3. Retrieval Strategy Selection & Comparative Benchmarking

Rather than guessing the best retrieval strategy, the system evaluates four architectures using a dedicated evaluation harness (`evals/retrieval_strategies.py`):

```text
                   ┌───────────────────────────────┐
                   │  Input Transcript & Question  │
                   └───────────────┬───────────────┘
                                   │
        ┌──────────────────┬───────┴───────┬──────────────────┐
        ▼                  ▼               ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Simple Fixed  │  │   Semantic    │  │ Hybrid Search │  │ FlashRank     │
│   Chunking    │  │   Chunking    │  │ (BM25 + Dense)│  │ Cross-Encoder │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │               │                  │
        └──────────────────┼───────────────┴──────────────────┘
                           ▼
               ┌─────────────────────────┐
               │ Reciprocal Rank Fusion  │
               │ & Cross-Encoder Scoring │
               └────────────┬────────────┘
                            ▼
               ┌─────────────────────────┐
               │ Grounded RAG Generation │
               └─────────────────────────┘
```

| Strategy                    | Implementation Details                                        | Best Used For                                       | Trade-offs                                    |
| --------------------------- | ------------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------- |
| **1. Simple Chunking**      | 500 characters, 50 character overlap                          | Uniform, dense discussions                          | Can sever thoughts across sentence boundaries |
| **2. Semantic Chunking**    | Percentile distance shift across sentence vectors             | Unstructured discussions with frequent topic shifts | Higher index-time computation                 |
| **3. Hybrid Search**        | Dense MiniLM + Sparse BM25 Okapi fused via RRF                | Technical jargon, exact names, timestamps, acronyms | Requires maintaining dual indexing structures |
| **4. Contextual Reranking** | Top-8 dense candidates re-scored via `FlashRank` (`TinyBERT`) | Complex questions requiring deep relevance scoring  | Added ~30ms cross-encoder inference step      |

### Reciprocal Rank Fusion (RRF) Formulation

$$ \text{RRF Score}(d) = \sum_{m \in M} \frac{1}{60 + \text{Rank}_m(d)} $$

---

## 4. Evaluation Framework: Ragas Metric Selection

| Metric                | Target Threshold | Evaluated Failure Mode                                                  |
| --------------------- | :--------------: | ----------------------------------------------------------------------- |
| **Faithfulness**      |    $\ge 0.85$    | Model hallucinating facts not grounded in retrieved transcript chunks   |
| **Context Recall**    |    $\ge 0.80$    | Retriever omitting chunks necessary to answer the ground-truth question |
| **Context Precision** |    $\ge 0.80$    | Retriever injecting noisy, irrelevant context that degrades generation  |

### Rationale

LLM-as-a-judge evaluation via Ragas isolates retrieval failures from generation failures. If Faithfulness drops, the issue lies in prompt grounding or LLM temperature; if Context Recall drops, the retriever strategy or chunk size requires calibration.

---

## 5. CI/CD Architecture Quality Gate (GitHub Actions)

| Design Choice           | Implementation                                  | Purpose                                                  |
| ----------------------- | ----------------------------------------------- | -------------------------------------------------------- |
| **Trigger Mechanism**   | `on: [push, pull_request]` on `main` / `master` | Ensures broken models cannot be merged                   |
| **Runner Optimization** | PyTorch CPU-only wheel installation             | Reduces CI workflow run time from ~21 mins to < 2 mins   |
| **Threshold Gate**      | `sys.exit(1)` if `min_faithfulness < 0.85`      | Automated build blocker                                  |
| **Secret Management**   | GitHub Actions Encrypted Secrets                | Secure injection of `MISTRAL_API_KEY` and `GROQ_API_KEY` |

### Rationale

Automated regression testing prevents changes to prompts, splitters, or models from silently degrading answer accuracy. The pipeline runs headlessly on every commit, validating system performance against golden datasets before deployment.

---

## 6. Prompt Engineering & Anti-Hallucination Guardrails

### System Prompt Structure

```text
You are an expert assistant for meeting intelligence.
Answer the question based ONLY on the following context:

{context}

If the information is not contained within the context, respond with:
"I could not find this information in the meeting transcript."
Do NOT extrapolate, infer, or hallucinate details.
```

### Rationale

* **Zero Speculation:** For business meetings and technical reviews, an explicit statement of omission is vastly superior to a plausible hallucination.
* **Deterministic Fallback:** The deterministic rejection string allows evaluation scripts to clearly identify missing retrieval context without ambiguous LLM answers.
