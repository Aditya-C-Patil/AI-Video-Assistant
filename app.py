import os
import time
from dotenv import load_dotenv
import streamlit as st

from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import ask_question, build_rag_chain
from core.summarizer import generate_title, summarize
from core.transcriber import transcribe_all
from utils.audio_processor import process_input

load_dotenv()

# ─── Configuration & Theme ───────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS AI // Multi-Stage Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #05070e;
    --surface: rgba(13, 17, 30, 0.75);
    --border: rgba(99, 102, 241, 0.2);
    --border-bright: rgba(99, 102, 241, 0.5);
    --primary: #6366f1;
    --cyan: #06b6d4;
    --emerald: #10b981;
    --text-high: #f8fafc;
    --text-mid: #94a3b8;
}

* { font-family: 'Inter', sans-serif; }
h1, h2, h3, .brand-title { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }
code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

.stApp { background-color: var(--bg) !important; color: var(--text-high) !important; }

/* Dynamic Ambient Glow */
.stApp::before {
    content: '';
    position: fixed;
    top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle at 80% 20%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 20% 80%, rgba(6, 182, 212, 0.08) 0%, transparent 35%);
    pointer-events: none; z-index: 0;
}

/* Glassmorphism Containers */
.glass-card {
    background: var(--surface);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 1.25rem;
    position: relative;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.brand-header {
    background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 50%, var(--cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: clamp(2rem, 3.5vw, 2.8rem);
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.4rem;
}

.brand-sub {
    color: var(--text-mid);
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 1.5rem;
}

.chip {
    display: inline-flex;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.chip-primary { background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.3); }
.chip-emerald { background: rgba(16, 185, 129, 0.15); color: var(--emerald); border: 1px solid rgba(16, 185, 129, 0.3); }

.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 0.65rem 1.25rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
}

[data-testid="stSidebar"] {
    background: rgba(8, 11, 20, 0.95) !important;
    border-right: 1px solid var(--border) !important;
}

/* Chat bubble styling */
[data-testid="stChatMessage"] {
    background: rgba(13, 17, 30, 0.6) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(10px) !important;
    margin-bottom: 0.8rem !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-color: rgba(99, 102, 241, 0.35) !important;
    background: rgba(99, 102, 241, 0.08) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── State Machine Routing ───────────────────────────────────────────────────
if "current_page" not in st.session_state:
    groq_env = os.getenv("GROQ_API_KEY", "").strip()
    mistral_env = os.getenv("MISTRAL_API_KEY", "").strip()
    
    # Auto-route to INGEST on cold boot if .env exists
    if groq_env and mistral_env:
        st.session_state.current_page = "INGEST"
    else:
        st.session_state.current_page = "AUTH"

if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def navigate_to(page_name: str):
    st.session_state.current_page = page_name
    st.rerun()


# ─── Sidebar Navigation HUD ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="brand-title" style="font-size:1.4rem; font-weight:700; color:white;">⚡ NEXUS // OS</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="brand-sub">Neural Multi-Modal Suite</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    pages = [
        ("AUTH", "🔑 Credentials & Keys"),
        ("INGEST", "🚀 Ingest & Process"),
        ("INTELLIGENCE", "📊 Tactical Intelligence"),
        ("CHAT", "💬 Vector RAG Chat"),
    ]

    for page_key, label in pages:
        is_active = st.session_state.current_page == page_key
        prefix = "👉 " if is_active else "   "

        # Lock results-dependent pages if no video has been processed yet
        disabled = False
        if page_key in ["INTELLIGENCE", "CHAT"] and not st.session_state.result:
            disabled = True

        if st.button(
            f"{prefix}{label}",
            key=f"nav_{page_key}",
            disabled=disabled,
            use_container_width=True,
        ):
            navigate_to(page_key)

    if st.session_state.result:
        st.markdown("---")
        st.markdown(
            f"""
        <div style="font-size: 0.75rem; color: var(--text-mid); line-height: 1.6;">
            <div>• <b>Title:</b> {st.session_state.result['title'][:20]}...</div>
            <div>• <b>Words:</b> {st.session_state.result['word_count']}</div>
            <div>• <b>Chunks:</b> {st.session_state.result['chunks_count']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: CREDENTIALS & INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.current_page == "AUTH":
    st.markdown(
        '<div class="brand-header">System Credentials</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="brand-sub">Neural Engine & API Configuration</div>',
        unsafe_allow_html=True,
    )

    groq_stored = os.getenv("GROQ_API_KEY", "")
    mistral_stored = os.getenv("MISTRAL_API_KEY", "")
    has_valid_env = bool(groq_stored.strip() and mistral_stored.strip())

    if has_valid_env:
        st.markdown(
            """
        <div class="glass-card" style="border-left: 3px solid var(--emerald);">
            <div style="color: var(--emerald); font-weight: 600; font-size: 0.85rem; margin-bottom: 0.3rem;">
                ACTIVE CONFIGURATION DETECTED
            </div>
            <div style="color: var(--text-mid); font-size: 0.85rem;">
                Environment credentials have been automatically loaded from your <code>.env</code> file. You can view or override them below.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        groq_val = st.text_input(
            "Groq API Key (Whisper LPU Inference)",
            value=groq_stored,
            type="password",
            placeholder="gsk_...",
            help="Used for fast Hindi/Hinglish to English Whisper-large-v3 translation.",
        )
    with col2:
        mistral_val = st.text_input(
            "Mistral AI Key (RAG Reasoning)",
            value=mistral_stored,
            type="password",
            placeholder="...",
            help="Used for document synthesis and question answering.",
        )

    st.write("")
    btn_col1, btn_col2 = st.columns([2, 5])
    with btn_col1:
        save_btn = st.button("💾 SAVE & CONTINUE →", use_container_width=True)

    if save_btn:
        if groq_val.strip() and mistral_val.strip():
            os.environ["GROQ_API_KEY"] = groq_val.strip()
            os.environ["MISTRAL_API_KEY"] = mistral_val.strip()
            st.success("Credentials saved successfully!")
            time.sleep(0.3)
            navigate_to("INGEST")
        else:
            st.error("Please supply both API keys before continuing.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: INGESTION & PIPELINE COMPILATION
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "INGEST":
    st.markdown(
        '<div class="brand-header">Video Ingestion Engine</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="brand-sub">Transcribe, Translate & Vectorize Streams</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="glass-card">
        <div style="font-size: 0.9rem; color: var(--text-high); margin-bottom: 0.5rem;">
            Enter any <b>YouTube URL</b> or <b>local file path</b> (.mp4, .mp3, .wav).
            Hindi/Hinglish audio is automatically translated to English embeddings via Groq LPUs.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    source_path = st.text_input(
        "Source Media URL / Local Path",
        placeholder="https://youtu.be/... or D:/recordings/meeting.mp4",
    )

    start_btn = st.button("🚀 COMPILE NEURAL PIPELINE", use_container_width=True)

    if start_btn:
        if not source_path.strip():
            st.error("Please provide a valid source path or URL.")
        else:
            status_box = st.status(
                "⚡ Ingesting & processing audio...", expanded=True
            )
            try:
                t0 = time.time()
                with status_box:
                    st.write("🔊 Slicing audio into compressed streams...")
                    chunks = process_input(source_path.strip())

                    st.write(
                        "⚡ Transcribing via Groq Whisper-Large-v3 clusters..."
                    )
                    transcript = transcribe_all(chunks)

                    st.write(
                        "🧠 Embedding chunks and building ChromaDB Vector Store..."
                    )
                    rag_chain = build_rag_chain(transcript)

                    st.write("📋 Generating Executive Synthesis...")
                    title = generate_title(transcript)
                    summary = summarize(transcript)

                    st.write("🎯 Extracting Decisions, Tasks, and Questions...")
                    action_items = extract_action_items(transcript)
                    decisions = extract_key_decisions(transcript)
                    questions = extract_questions(transcript)

                    elapsed = round(time.time() - t0, 2)
                    status_box.update(
                        label=f"✅ Video Compiled in {elapsed}s!",
                        state="complete",
                        expanded=False,
                    )

                st.session_state.result = {
                    "title": title,
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
                navigate_to("INTELLIGENCE")

            except Exception as e:
                status_box.update(
                    label=f"❌ Execution Failure: {e}",
                    state="error",
                    expanded=True,
                )
                st.error(str(e))

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: TACTICAL INTELLIGENCE & TRANSCRIPT
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "INTELLIGENCE":
    r = st.session_state.result
    st.markdown(
        f'<div class="brand-header">{r["title"]}</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="brand-sub">TACTICAL REPORT & STRUCTURED EXTRACTION</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("💬 Launch Vector RAG Chat →", use_container_width=True):
            navigate_to("CHAT")
    with c2:
        if st.button("📥 Ingest Another Video", use_container_width=True):
            navigate_to("INGEST")

    st.markdown("---")

    # Summary Card
    st.markdown(
        f"""
    <div class="glass-card">
        <span class="chip chip-primary" style="margin-bottom: 0.75rem;">Executive Summary</span>
        <div style="font-size: 0.95rem; line-height: 1.8; color: var(--text-high);">{r['summary']}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 3-Column Extraction Grid
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            f"""
        <div class="glass-card" style="height: 100%;">
            <span class="chip chip-emerald" style="margin-bottom: 0.75rem;">Action Items</span>
            <div style="font-size: 0.85rem; line-height: 1.7;">{r['action_items']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"""
        <div class="glass-card" style="height: 100%;">
            <span class="chip chip-primary" style="margin-bottom: 0.75rem;">Decisions</span>
            <div style="font-size: 0.85rem; line-height: 1.7;">{r['key_decisions']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            f"""
        <div class="glass-card" style="height: 100%;">
            <span class="chip chip-primary" style="margin-bottom: 0.75rem;">Open Questions</span>
            <div style="font-size: 0.85rem; line-height: 1.7;">{r['open_questions']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.write("")
    with st.expander("📜 View Master English Transcript"):
        st.text_area(
            "Transcript",
            value=r["transcript"],
            height=300,
            label_visibility="collapsed",
        )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4: NEURAL VECTOR RAG CHAT
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "CHAT":
    r = st.session_state.result
    st.markdown(
        f'<div class="brand-header">Meeting Vector RAG</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="brand-sub">Grounded In: {r["title"]}</div>',
        unsafe_allow_html=True,
    )

    if st.button("← Back to Intelligence Brief"):
        navigate_to("INTELLIGENCE")

    st.markdown("---")

    # Render previous conversation
    for msg in st.session_state.chat_history:
        avatar = "⚡" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Chat Input
    if query := st.chat_input("Ask any question regarding the transcript..."):
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Querying ChromaDB vector embeddings..."):
                answer = ask_question(r["rag_chain"], query)
                st.markdown(answer)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )