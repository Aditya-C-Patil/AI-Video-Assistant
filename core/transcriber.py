import os
import re
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Phrases Whisper emits on pure background noise or end credits
HALLUCINATION_EXACT_MATCHES = [
    "thank you for watching",
    "thanks for watching",
    "subtitles by",
    "subscribe to the channel",
    "please subscribe",
]


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment or .env file.")
    return Groq(api_key=api_key)


def is_hallucinated_noise(text: str) -> bool:
    """Detects whether a transcript chunk is empty, purely music/applause tags,

    or stuck in a repetitive token loop.
    """
    cleaned = text.lower().strip()
    if not cleaned or len(cleaned) < 5:
        return True

    # Strip bracketed audio metadata like [Music], (Applause), [Silence]
    stripped_metadata = re.sub(
        r"\[(music|applause|cheering|laughter|silence)\]|\((music|applause|cheering|laughter|silence)\)",
        "",
        cleaned,
    ).strip()
    if not stripped_metadata:
        return True

    # Check for exact subtitle/end-card hallucination phrases
    if any(cleaned == phrase for phrase in HALLUCINATION_EXACT_MATCHES):
        return True

    words = re.findall(r"\b[a-zA-Z0-9]+\b", cleaned)
    if not words:
        return True

    # Catch severe token loops (e.g., repeating the exact same word 20+ times)
    if len(words) > 15:
        diversity_ratio = len(set(words)) / len(words)
        if diversity_ratio < 0.12:
            return True

    return False


def transcribe_to_english(audio_path: str) -> str:
    client = get_groq_client()
    with open(audio_path, "rb") as file:
        response = client.audio.translations.create(
            file=(Path(audio_path).name, file.read()),
            model="whisper-large-v3",
            response_format="text",
            temperature=0.0,
        )
    return response.strip()


def transcribe_all(chunks: list) -> str:
    valid_chunks = []

    for i, chunk in enumerate(chunks):
        try:
            print(f"  → Processing chunk {i + 1}/{len(chunks)}: {chunk}")
            raw_text = transcribe_to_english(chunk)
            print(f"    Raw text: '{raw_text}'")

            if not is_hallucinated_noise(raw_text):
                valid_chunks.append(raw_text)
            else:
                print(f"    [Filtered] Non-speech noise detected in chunk {i + 1}")
        except Exception as e:
            print(f"Error processing chunk {chunk}: {e}")
        finally:
            if os.path.exists(chunk):
                try:
                    os.unlink(chunk)
                except OSError:
                    pass

    if not valid_chunks:
        return (
            "NO_SPEECH_DETECTED: No discernible spoken conversation was found"
            " in this audio stream."
        )

    return " ".join(valid_chunks).strip()
