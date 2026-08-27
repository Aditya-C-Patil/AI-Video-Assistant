import os
import re
import shutil
import tempfile
import time
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import ask_question, build_rag_chain, load_rag_chain
from core.summarizer import generate_title, summarize
from core.transcriber import transcribe_all
from core.vector_store import (
    delete_vector_store_collection,
    load_vector_store,
)
from utils.audio_processor import process_input

load_dotenv()

# ─── Streamlit Configuration & Dark-Neon Aesthetic ────────────────────────────
st.set_page_config(
    page_title="NEXUS AI // Meeting Intelligence OS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg: #030712;
    --surface: rgba(15, 23, 42, 0.7);
    --surface-accent: rgba(30, 41, 59, 0.7);
    --border: rgba(99, 102, 241, 0.25);
    --border-glow: rgba(99, 102, 241, 0.6);
    --primary: #6366f1;
    --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    --cyan: #06b6d4;
    --emerald: #10b981;
    --amber: #f59e0b;
    --rose: #f43f5e;
    --text-high: #f8fafc;
    --text-mid: #94a3b8;
    --text-muted: #64748b;
}

* { font-family: 'Plus Jakarta Sans', sans-serif; }
code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

.stApp {
    background-color: var(--bg) !important;
    color: var(--text-high) !important;
}

/* Ambient Radial Glow Background */
.stApp::before {
    content: '';
    position: fixed;
    top: -40%; left: -30%; width: 180%; height: 180%;
    background: radial-gradient(circle at 75% 25%, rgba(99, 102, 241, 0.14) 0%, transparent 45%),
                radial-gradient(circle at 20% 75%, rgba(6, 182, 212, 0.10) 0%, transparent 40%),
                radial-gradient(circle at 50% 50%, rgba(168, 85, 247, 0.06) 0%, transparent 50%);
    pointer-events: none; z-index: 0;
}

/* Futuristic Glass Card */
.glass-card {
    background: var(--surface);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 12px 35px -5px rgba(99, 102, 241, 0.25);
}

.hero-title {
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 40%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: clamp(2.2rem, 4vw, 3.2rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}

.hero-subtitle {
    color: var(--cyan);
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 1.75rem;
}

/* Status Chips */
.chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}
.chip-indigo { background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.35); }
.chip-cyan { background: rgba(6, 182, 212, 0.15); color: #67e8f9; border: 1px solid rgba(6, 182, 212, 0.35); }
.chip-emerald { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.35); }
.chip-amber { background: rgba(245, 158, 11, 0.15); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.35); }

/* Primary Button Styling */
.stButton > button {
    background: var(--primary-gradient) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    padding: 0.7rem 1.4rem !important;
    letter-spacing: 0.04em !important;
    font-size: 0.88rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 25px rgba(168, 85, 247, 0.5) !important;
}

/* Metric Display Cards */
.stat-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
}
.stat-value {
    font-size: 1.5rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #f8fafc;
}
.stat-label {
    font-size: 0.72rem;
    color: var(--text-mid);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

[data-testid="stSidebar"] {
    background: rgba(10, 15, 29, 0.95) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stChatMessage"] {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
    margin-bottom: 0.85rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-color: rgba(168, 85, 247, 0.4) !important;
    background: rgba(99, 102, 241, 0.09) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─── Utility & Disk Scrubbing Functions ───────────────────────────────────────
def sanitize_meeting_id(title: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", title.strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    date_tag = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{clean[:32]}_{date_tag}" if clean else f"session_{date_tag}"


def cleanup_directory(folder_path: str = "downloads"):
    """Thoroughly deletes all dangling files and temp chunks in the target directory."""
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Directory scrub warning ({item_path}): {e}")


def get_existing_collections() -> list:
    """Lists all active meeting collections stored in ChromaDB."""
    try:
        vs = load_vector_store("default_meeting")
        collections = vs._client.list_collections()
        return [
            c.name
            for c in collections
            if c.name not in ["default_meeting", "meeting_transcript"]
        ]
    except Exception:
        return []


# Run initial disk scrub to avoid leftover artifacts from prior sessions
cleanup_directory("downloads")

# ─── Session State Machine ───────────────────────────────────────────────────
if "current_page" not in st.session_state:
    groq_env = os.getenv("GROQ_API_KEY", "").strip()
    mistral_env = os.getenv("MISTRAL_API_KEY", "").strip()
    st.session_state.current_page = (
        "INGEST" if (groq_env and mistral_env) else "AUTH"
    )

if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "run_pipeline" not in st.session_state:
    st.session_state.run_pipeline = False
if "source_to_process" not in st.session_state:
    st.session_state.source_to_process = ""


def navigate_to(page_name: str):
    st.session_state.current_page = page_name
    st.rerun()


# ─── Sidebar Navigation & Session HUD ─────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-size:1.45rem; font-weight:800; color:white; letter-spacing:-0.03em;">⚡ NEXUS // OS</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-subtitle" style="margin-bottom:1.2rem;">Multimodal Meeting AI</div>',
        unsafe_allow_html=True,
    )

    pages = [
        ("AUTH", "🔑 Credentials & Keys"),
        ("INGEST", "🚀 Ingestion Hub"),
        ("INTELLIGENCE", "📊 Tactical Intelligence"),
        ("CHAT", "💬 Neural Vector RAG"),
    ]

    for page_key, label in pages:
        is_active = st.session_state.current_page == page_key
        prefix = "⚡ " if is_active else "   "
        disabled = (
            page_key in ["INTELLIGENCE", "CHAT"]
            and not st.session_state.result
        )

        if st.button(
            f"{prefix}{label}",
            key=f"nav_{page_key}",
            disabled=disabled,
            use_container_width=True,
        ):
            navigate_to(page_key)

    # Past Sessions & ChromaDB Multi-Tenancy Management
    existing_meetings = get_existing_collections()
    if existing_meetings:
        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.75rem; color:var(--cyan); font-family:JetBrains Mono; margin-bottom:0.5rem;">📁 PERSISTENT VECTOR STORES</div>',
            unsafe_allow_html=True,
        )
        selected_past = st.selectbox(
            "Select Collection",
            options=["-- Select Saved Meeting --"] + existing_meetings,
            label_visibility="collapsed",
        )

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if (
                st.button("🔄 Connect", use_container_width=True)
                and selected_past != "-- Select Saved Meeting --"
            ):
                chain = load_rag_chain(meeting_id=selected_past)
                st.session_state.result = {
                    "title": selected_past.replace("_", " ").upper(),
                    "meeting_id": selected_past,
                    "transcript": "(Transcript indexed in ChromaDB vector collection)",
                    "summary": "Collection reconnected from persistent vector storage.",
                    "action_items": "Ready for retrieval in Neural Vector RAG chat session.",
                    "key_decisions": "Ready for retrieval in Neural Vector RAG chat session.",
                    "open_questions": "Ready for retrieval in Neural Vector RAG chat session.",
                    "rag_chain": chain,
                    "chunks_count": "Indexed",
                    "word_count": "Persistent",
                    "elapsed_time": 0.0,
                }
                st.session_state.chat_history = []
                navigate_to("CHAT")

        with btn_c2:
            if (
                st.button("🗑️ Delete", use_container_width=True)
                and selected_past != "-- Select Saved Meeting --"
            ):
                delete_vector_store_collection(selected_past)
                if (
                    st.session_state.result
                    and st.session_state.result.get("meeting_id")
                    == selected_past
                ):
                    st.session_state.result = None
                    st.session_state.chat_history = []
                st.toast(f"Deleted collection: {selected_past}")
                time.sleep(0.3)
                st.rerun()

    # Active Session Telemetry
    if st.session_state.result:
        r = st.session_state.result
        st.markdown("---")
        st.markdown(
            f"""
        <div style="font-size: 0.76rem; color: var(--text-mid); line-height: 1.8;">
            <div>• <b>Meeting:</b> <span style="color:white;">{r['title'][:16]}...</span></div>
            <div>• <b>Collection:</b> <span class="mono" style="color:var(--cyan);">{r.get('meeting_id', 'active')}</span></div>
            <div>• <b>Words:</b> <span class="mono">{r['word_count']}</span></div>
            <div>• <b>Latency:</b> <span class="mono">{r['elapsed_time']}s</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════════════════════════════════════
# VIEW 1: AUTH & CREDENTIALS
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.current_page == "AUTH":
    st.markdown(
        '<div class="hero-title">System Credentials</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-subtitle">High-Speed LPU & Model Reasoning Engines</div>',
        unsafe_allow_html=True,
    )

    groq_stored = os.getenv("GROQ_API_KEY", "")
    mistral_stored = os.getenv("MISTRAL_API_KEY", "")

    st.markdown(
        """
    <div class="glass-card">
        <div style="font-size:0.9rem; color:var(--text-high); line-height:1.6;">
            Configure your enterprise API endpoints. Keys provided below are held securely in application runtime memory.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        groq_val = st.text_input(
            "Groq API Key (Whisper-large-v3 Transcription)",
            value=groq_stored,
            type="password",
            placeholder="gsk_...",
        )
    with col2:
        mistral_val = st.text_input(
            "Mistral AI Key (RAG Reasoning & Synthesis)",
            value=mistral_stored,
            type="password",
            placeholder="...",
        )

    st.write("")
    btn_col, _ = st.columns([1, 2])
    with btn_col:
        if st.button("💾 SAVE & INITIALIZE PIPELINE →", use_container_width=True):
            if groq_val.strip() and mistral_val.strip():
                os.environ["GROQ_API_KEY"] = groq_val.strip()
                os.environ["MISTRAL_API_KEY"] = mistral_val.strip()
                st.toast("Credentials verified and saved!")
                time.sleep(0.3)
                navigate_to("INGEST")
            else:
                st.error("Please supply both API keys.")

# ═════════════════════════════════════════════════════════════════════════════
# VIEW 2: INGESTION HUB (URL + LOCAL FILE DRAG-AND-DROP)
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "INGEST":
    st.markdown(
        '<div class="hero-title">Media Ingestion Engine</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-subtitle">Transcribe, Partition & Vectorize Audio Streams</div>',
        unsafe_allow_html=True,
    )

    tab_url, tab_file = st.tabs(
        ["🌐 YouTube / Remote Stream", "📁 Local Media Upload"]
    )

    source_selected = None

    with tab_url:
        st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
        url_input = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
            key="input_url_field",
        )
        if st.button(
            "🚀 PROCESS YOUTUBE STREAM",
            key="btn_proc_url",
            use_container_width=True,
        ):
            if url_input.strip():
                source_selected = url_input.strip()
            else:
                st.error("Please provide a valid YouTube URL.")

    with tab_file:
        st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Audio / Video Meeting Recording",
            type=["mp4", "mov", "avi", "m4a", "mp3", "wav", "mkv"],
            help="Supports high-definition video and compressed audio formats.",
        )
        if st.button(
            "🚀 PROCESS UPLOADED MEDIA",
            key="btn_proc_file",
            use_container_width=True,
        ):
            if uploaded_file is not None:
                temp_dir = os.path.join(tempfile.gettempdir(), "nexus_uploads")
                os.makedirs(temp_dir, exist_ok=True)
                temp_media_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_media_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                source_selected = temp_media_path
            else:
                st.error("Please drag-and-drop a media file first.")

    if source_selected:
        st.session_state.source_to_process = source_selected
        st.session_state.run_pipeline = True
        st.rerun()

    # Dedicated Pipeline Execution Coordinator
    if st.session_state.run_pipeline:
        source = st.session_state.source_to_process
        status_box = st.status(
            "⚡ Compiling Neural Multimodal Pipeline...", expanded=True
        )

        try:
            t0 = time.time()
            with status_box:
                st.write(
                    "🔊 Slicing audio into 16kHz mono chunks and normalizing..."
                )
                cleanup_directory("downloads")
                chunks = process_input(source)

                st.write(
                    "⚡ Performing high-speed Whisper-large-v3 transcription via Groq LPU..."
                )
                transcript = transcribe_all(chunks)

                # Thorough post-transcription disk cleanup
                cleanup_directory("downloads")
                if (
                    isinstance(source, str)
                    and os.path.exists(source)
                    and "nexus_uploads" in source
                ):
                    try:
                        os.unlink(source)
                    except OSError:
                        pass

                if (
                    not transcript
                    or transcript.startswith("NO_SPEECH_DETECTED")
                    or len(transcript.strip()) < 15
                ):
                    status_box.update(
                        label="⚠️ No Speech Detected",
                        state="error",
                        expanded=True,
                    )
                    st.warning(
                        "No spoken conversation was detected in this media file."
                    )
                    st.session_state.run_pipeline = False
                else:
                    st.write("📋 Synthesizing executive meeting brief...")
                    title = generate_title(transcript)
                    summary = summarize(transcript)

                    meeting_id = sanitize_meeting_id(title)
                    st.write(
                        f"🧠 Embedding chunks into ChromaDB collection: `{meeting_id}`..."
                    )
                    rag_chain = build_rag_chain(
                        transcript, meeting_id=meeting_id
                    )

                    st.write(
                        "🎯 Extracting decisions, action items, and open questions..."
                    )
                    action_items = extract_action_items(transcript)
                    decisions = extract_key_decisions(transcript)
                    questions = extract_questions(transcript)

                    elapsed = round(time.time() - t0, 2)
                    status_box.update(
                        label=f"✅ Pipeline Completed in {elapsed}s!",
                        state="complete",
                        expanded=False,
                    )

                    st.session_state.result = {
                        "title": title,
                        "meeting_id": meeting_id,
                        "transcript": transcript,
                        "summary": summary,
                        "action_items": action_items,
                        "key_decisions": decisions,
                        "open_questions": questions,
                        "rag_chain": rag_chain,
                        "chunks_count": len(chunks),
                        "word_count": len(transcript.split()),
                        "elapsed_time": elapsed,
                    }
                    st.session_state.chat_history = []
                    st.session_state.run_pipeline = False
                    navigate_to("INTELLIGENCE")

        except Exception as e:
            cleanup_directory("downloads")
            status_box.update(
                label=f"❌ Pipeline Failed: {e}", state="error", expanded=True
            )
            st.session_state.run_pipeline = False
            st.error(str(e))

# ═════════════════════════════════════════════════════════════════════════════
# VIEW 3: TACTICAL INTELLIGENCE BRIEF
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "INTELLIGENCE":
    r = st.session_state.result
    st.markdown(
        f'<div class="hero-title">{r["title"]}</div>', unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="hero-subtitle">COLLECTION: {r.get("meeting_id", "ACTIVE")} // LATENCY: {r["elapsed_time"]}s</div>',
        unsafe_allow_html=True,
    )

    # Telemetry KPI Ribbon
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f'<div class="stat-box"><div class="stat-value">{r["word_count"]}</div><div class="stat-label">Words Ingested</div></div>',
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f'<div class="stat-box"><div class="stat-value">{r["chunks_count"]}</div><div class="stat-label">Audio Chunks</div></div>',
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f'<div class="stat-box"><div class="stat-value">{r["elapsed_time"]}s</div><div class="stat-label">Processing Time</div></div>',
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            '<div class="stat-box"><div class="stat-value" style="color:var(--emerald);">READY</div><div class="stat-label">Vector Index Status</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    nav_c1, nav_c2 = st.columns(2)
    with nav_c1:
        if st.button(
            "💬 LAUNCH VECTOR RAG CHAT →",
            key="btn_to_chat",
            use_container_width=True,
        ):
            navigate_to("CHAT")
    with nav_c2:
        if st.button(
            "📥 INGEST ANOTHER RECORDING",
            key="btn_new_ingest",
            use_container_width=True,
        ):
            navigate_to("INGEST")

    st.markdown("---")

    # Executive Brief Card
    st.markdown(
        f"""
    <div class="glass-card">
        <span class="chip chip-indigo" style="margin-bottom: 0.85rem;">Executive Synthesis</span>
        <div style="font-size: 0.96rem; line-height: 1.85; color: var(--text-high);">{r['summary']}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 3-Column Structured Information Grid
    c_act, c_dec, c_qst = st.columns(3)
    with c_act:
        st.markdown(
            f"""
        <div class="glass-card" style="height: 100%;">
            <span class="chip chip-emerald" style="margin-bottom: 0.85rem;">Action Items</span>
            <div style="font-size: 0.88rem; line-height: 1.75; color: #e2e8f0;">{r['action_items']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c_dec:
        st.markdown(
            f"""
        <div class="glass-card" style="height: 100%;">
            <span class="chip chip-cyan" style="margin-bottom: 0.85rem;">Key Decisions</span>
            <div style="font-size: 0.88rem; line-height: 1.75; color: #e2e8f0;">{r['key_decisions']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c_qst:
        st.markdown(
            f"""
        <div class="glass-card" style="height: 100%;">
            <span class="chip chip-amber" style="margin-bottom: 0.85rem;">Open Questions</span>
            <div style="font-size: 0.88rem; line-height: 1.75; color: #e2e8f0;">{r['open_questions']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.write("")
    with st.expander("📜 View Master Transcript"):
        st.text_area(
            "Master Transcript",
            value=r["transcript"],
            height=320,
            label_visibility="collapsed",
        )

# ═════════════════════════════════════════════════════════════════════════════
# VIEW 4: NEURAL VECTOR RAG CHAT INTERFACE
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "CHAT":
    r = st.session_state.result
    st.markdown(
        '<div class="hero-title">Neural Vector RAG</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="hero-subtitle">GROUNDED IN: {r["title"]} // COLLECTION: {r.get("meeting_id", "ACTIVE")}</div>',
        unsafe_allow_html=True,
    )

    if st.button("← Back to Tactical Intelligence"):
        navigate_to("INTELLIGENCE")

    st.markdown("---")

    # Render Persistent Chat Logs
    for msg in st.session_state.chat_history:
        avatar = "⚡" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Query Input Box
    if query := st.chat_input("Ask any question regarding the transcript..."):
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Retrieving from ChromaDB & generating response..."):
                answer = ask_question(r["rag_chain"], query)
                st.markdown(answer)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
