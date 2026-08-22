# ⚡ NEXUS AI // Intelligent Video Meeting Assistant & RAG Platform

NEXUS AI is an intelligent video assistant that transforms raw video and audio streams into actionable intelligence. By combining **Groq LPU transcription (`whisper-large-v3`)**, **ChromaDB vector indexing**, and **Mistral AI reasoning**, it seamlessly translates Hindi/Hinglish audio into English, extracts key decisions and action items, and enables real-time, grounded RAG chat over long-form meeting recordings and YouTube videos.

---

## 📑 Architectural Flow

```text
    [ YouTube URL / Local MP4 / MP3 ]
                  │
                  ▼
    [ utils/audio_processor.py ]
                  │
                  ├──► FFmpeg Normalization
                  └──► 64k Audio Slicing
                  │
                  ▼
    [ core/transcriber.py ]
                  │
                  └──► Groq LPU whisper-large-v3
                       Speech → English Text
                  │
          ┌───────┴───────────────────────────┐
          ▼                                   ▼
[ core/vector_store.py ]        [ core/summarizer.py ]
          │                     [ core/extractor.py ]
          ▼                                   │
   (ChromaDB Indexing)                        ▼
                                      (Executive Brief,
                                       Tasks, Decisions, Q&A)
          │
          └───────────────┬───────────────────┘
                          ▼
              [ core/rag_engine.py ]
                          │
                          ▼
              (LangChain LCEL Pipeline
                  + ChatMistralAI)
                          │
                          ▼
              [ Streamlit Multi-Stage UI
                       (app.py)
                    / CLI (main.py) ]
```

---

## 📂 Project Structure & Module Breakdown

```text
AI Video Assistant/
│
├── core/                           # Core AI and Retrieval Pipeline
│   ├── extractor.py                # Extracts action items, key decisions, and open questions
│   ├── rag_engine.py               # Assembles LCEL RAG chain and handles vector search QA
│   ├── summarizer.py               # Generates session titles and executive summaries
│   ├── transcriber.py              # Groq whisper-large-v3 speech-to-English translation
│   └── vector_store.py             # Document chunking, HuggingFace embeddings, ChromaDB store
│
├── utils/                          # Audio Processing & Media Tools
│   └── audio_processor.py          # yt-dlp download, FFmpeg slicing, and format conversions
│
├── app.py                          # Modern multi-stage Streamlit Web GUI
├── main.py                         # Standalone CLI execution pipeline
├── requirements.txt                # Pinned dependencies
├── .env.example                    # Example to setup API keys & environment secrets
└── .gitignore                      # Git exclusion rules
```

---

## ⚡ Key Features

* **Zero-Compute Translation**
  Translates Hindi and Hinglish speech directly into English text in a single inference call via Groq-hosted `whisper-large-v3`.

* **Grounded LCEL RAG**
  Strict meeting question answering powered by `ChatMistralAI`, preventing hallucinations by anchoring context exclusively to ChromaDB vector search results.

* **Tactical Intelligence Extraction**
  Automated extraction of structured executive summaries, prioritized action items, key decisions, and unresolved questions.

* **Resilient Audio Pipeline**
  Automated compression, 64k bitrate export, and payload chunking to prevent API timeout and size-limit errors.

* **Multi-Stage Cyberpunk GUI**
  A Streamlit interface equipped with automated `.env` verification, progress telemetry, structured metric HUDs, and interactive vector chat.

---

## 🛠️ Quickstart & Prerequisites

### 1. System Dependency

**FFmpeg** — Ensure FFmpeg is installed and registered in your system environment `PATH`.

### 2. Installation & Virtual Environment Setup

```bash
# Clone the repository
git clone https://github.com/Aditya-C-Patil/AI-Video-Assistant.git
cd AI-Video-Assistant

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install project dependencies
pip install -r requirements.txt
```

### 3. Environment Variables Configuration

Create a `.env` file in the root directory.

---

## 🚀 Running the Application

### Option A — Launch the Streamlit Web Application

```bash
streamlit run app.py
```

### Option B — Run the CLI Pipeline

```bash
python -u main.py
```
