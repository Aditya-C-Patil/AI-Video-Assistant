# ⚡ NEXUS AI // 🎙️ Video / Audio Meeting Intelligence & RAG System with CI/CD Eval Harness

[![RAG Evaluation Suite CI](https://github.com/Aditya-C-Patil/AI-Video-Assistant/actions/workflows/rag_eval.yml/badge.svg)](https://github.com/Aditya-C-Patil/AI-Video-Assistant/actions/workflows/rag_eval.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Evaluation Framework](https://img.shields.io/badge/Eval-RAGAS-orange.svg)](https://docs.ragas.io/)
[![Vector Store](https://img.shields.io/badge/VectorStore-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
---

NEXUS AI is an end-to-end multimodal meeting intelligence and Retrieval-Augmented Generation (RAG) platform. The system ingests long-form video/audio files or YouTube URLs, automates chunked transcription and structured summarization, and enables context-grounded Q&A with dynamic source citation.

Unlike basic RAG demonstrations, this repository implements a complete **Evaluation Harness & CI/CD Regression Pipeline** that systematically benchmarks 4 distinct retrieval architectures and blocks quality regressions on push using automated metric thresholds.

---

## 🏗️ System Architecture

```text
[ Video / Audio / YouTube ]
              │
              ▼
┌───────────────────────────────────────────────┐
│ 1. Ingestion, Preprocessing & Chunking        │
│    - Format standardization (16kHz mono WAV)  │
│    - Dynamic chunk slicing for API limits     │
│    - Automated temp cleanup & storage bounds  │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ 2. High-Fidelity Transcription                │
│    - Groq Whisper-large-v3                    │
│    - Sub-chunk timestamp & text alignment     │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ 3. Intelligent Analysis & Vector Indexing     │
│    - Executive summaries & action items       │
│    - Dense Embeddings (all-MiniLM-L6-v2)      │
│    - Dynamic ChromaDB Collections (by ID)     │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ 4. Grounded RAG Generation & QA Interface     │
│    - Anti-hallucination prompt constraints    │
│    - Mistral Large / Groq Llama-3             │
│    - Streamlit Dashboard & CLI Entry Point    │
└───────────────────────────────────────────────┘
```

---

## 🔬 RAG Evaluation Harness & CI/CD Pipeline

To ensure retrieval precision, hallucination prevention, and data-driven architectural selection, this project incorporates a **3-stage evaluation lifecycle**:

```text
┌───────────────────────────┐      ┌────────────────────────────┐      ┌───────────────────────────┐
│  1. Ground-Truth Dataset  │  ──► │  2. Multi-Method Benchmark │  ──► │  3. GitHub Actions CI/CD  │
│  - Golden Q&A Benchmarks  │      │  - Simple vs Semantic      │      │  - Automated Test Matrix  │
│  - Reference Contexts     │      │  - Hybrid vs Reranking     │      │  - Quality Gate (>= 0.85) │
└───────────────────────────┘      └────────────────────────────┘      └───────────────────────────┘
```

### 1. Ground-Truth Golden Dataset

Structured evaluation sets (`evals/benchmark_data.json`) containing domain questions, target ground-truth answers, and reference context spans to measure extraction accuracy and retrieval precision.

### 2. Multi-Paradigm Retrieval Benchmarking

The harness evaluates four retrieval strategies against the golden dataset:

1. **Simple Fixed Chunking:** Recursive character splitting (500 chars, 50 overlap) indexed in ChromaDB with dense vector search.
2. **Semantic Chunking:** Dynamic boundary detection using percentile distance shifts across consecutive sentence embeddings.
3. **Hybrid Search (Dense + Lexical BM25):** Combines dense vector similarity with sparse BM25 Okapi keyword scores using **Reciprocal Rank Fusion (RRF)**:

$$
\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{60 + \text{Rank}_m(d)}
$$

4. **Contextual Reranking:** High-recall dense candidate retrieval ($k=8$) re-scored through a local cross-encoder (`FlashRank` / `TinyBERT`) to isolate the top 3 most relevant passages.

### 3. Automated CI/CD Regression Testing

Every push and pull request triggers a GitHub Actions runner (`.github/workflows/rag_eval.yml`) that executes the comparative evaluation matrix. If average **Faithfulness** drops below **0.85**, the CI job fails, preventing regressions from merging into production.

#### Evaluation Benchmark Results

```text
════════════════════════════════════════════════════════════
🏆 RAG ARCHITECTURE EVALUATION MATRIX
════════════════════════════════════════════════════════════
                Method  Faithfulness  Context Recall  Context Precision
       Simple Chunking           1.0             1.0                1.0
     Semantic Chunking           1.0             1.0                1.0
         Hybrid Search           1.0             1.0                1.0
 Reranking (FlashRank)           1.0             1.0                1.0

✅ All RAG architectures passed minimum quality threshold (Faithfulness >= 0.85).
```

---

## ✨ Engineering Highlights

* **Multi-Source Ingestion:** Native handling of local media (`.mp4`, `.mov`, `.mp3`, `.wav`, `.m4a`) and direct YouTube audio extraction via `yt-dlp`.
* **Chunked Audio Pipeline & Resource Management:** Chunks audio into standardized segments via `pydub`/`ffmpeg` to respect upstream API limits, with automated post-processing disk cleanup utilities.
* **Meeting-Isolated Vector Indexes:** ChromaDB collections are dynamically isolated by unique `meeting_id` to prevent cross-session context bleeding and vector leakage.
* **Hallucination Guardrails:** RAG prompt templates strictly constrain responses to retrieved chunks, refusing out-of-context queries deterministically.
* **Cloud CI/CD Execution:** Optimized PyTorch CPU caching and environment shims for reliable, fast execution on GitHub Actions runners.

---

## 🛠️ Tech Stack

| **Component**              | **Technology**                                              |
| -------------------------- | ----------------------------------------------------------- |
| **LLMs & Generation**      | Mistral Large (`mistral-large-latest`), Groq Llama-3        |
| **Speech-to-Text**         | Groq Whisper-large-v3                                       |
| **Vector DB & Storage**    | ChromaDB (Collection-partitioned)                           |
| **Embeddings & Reranking** | HuggingFace `all-MiniLM-L6-v2`, BM25 Okapi, FlashRank       |
| **Orchestration**          | LangChain Core, LangChain Community, LangChain Experimental |
| **Evaluation Suite**       | Ragas, HuggingFace Datasets, Pandas, NumPy                  |
| **Audio Processing**       | FFmpeg, PyDub, yt-dlp                                       |
| **Interface & CLI**        | Streamlit, CLI entry point (`main.py`)                      |
| **CI/CD**                  | GitHub Actions (Ubuntu x86_64, CPU Wheel Optimization)      |

---

## 📁 Repository Structure

```text
├── .github/
│   └── workflows/
│       └── rag_eval.yml            # Automated CI/CD RAG evaluation workflow
├── core/
│   ├── audio_processor.py          # YouTube download, audio conversion, and slicing
│   ├── transcriber.py              # Whisper API client and chunked transcription
│   ├── meeting_analyzer.py         # Structured summary & action item generation
│   ├── vector_store.py             # Isolated ChromaDB vector indexing and retrieval
│   └── rag_engine.py               # LLM integration, prompt templates, and QA logic
├── evals/
│   ├── benchmark_data.json         # Golden evaluation benchmark dataset
│   ├── retrieval_strategies.py     # Implementations of Simple, Semantic, Hybrid, and Reranked RAG
│   ├── evaluate_rag.py             # Standalone single-pipeline evaluation script
│   ├── benchmark_methods.py        # Comparative matrix runner and threshold gate
│   └── results/                    # Evaluation artifacts and exported CSV scores
├── utils/
│   └── audio_cleanup.py            # Temporary file deletion and cleanup utilities
├── app.py                          # Streamlit interactive web dashboard
├── main.py                         # CLI entry point for full pipeline execution
├── .env.example                    # Template for required environment variables
├── requirements.txt                # Pinned Python package dependencies
├── LICENSE                         # Project license
└── README.md                       # Documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Aditya-C-Patil/AI-Video-Assistant.git
cd AI-Video-Assistant
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the template and provide your API keys:

```bash
cp .env.example .env
```

Inside `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 5. Run the Application

**Interactive UI:**

```bash
streamlit run app.py
```

**Command Line Pipeline:**

```bash
python main.py
```

---

## 🧪 Running the Evaluation Suite

**Run the 4-Method Comparison Matrix & CI Quality Gate:**

```bash
python evals/benchmark_methods.py
```

**Run Single-Pipeline RAG Evaluation:**

```bash
python evals/evaluate_rag.py
```
