import os
import re
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Patterns that Whisper creates when listening to music, silence, or noise
HALLUCINATION_PATTERNS = [
    r"\b(thank\s+you|thanks\s+for\s+watching|subtitles\s+by|please\s+subscribe|subscribe)\b",
    r"\b(music|applause|cheering|laughter)\b",
    r"\b(the|ah|um|uh|you)\b",
]


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment or .env file.")
    return Groq(api_key=api_key)


def is_hallucinated_noise(text: str) -> bool:
    """Detects whether a transcript chunk is just looped filler words or music artifacts."""
    cleaned = text.lower().strip()
    if not cleaned or len(cleaned) < 6:
        return True

    words = re.findall(r"\b[a-zA-Z]+\b", cleaned)
    if not words:
        return True

    unique_words = set(words)
    diversity_ratio = len(unique_words) / len(words)
    if len(words) > 4 and diversity_ratio < 0.35:
        return True

    stripped_text = cleaned
    for pattern in HALLUCINATION_PATTERNS:
        stripped_text = re.sub(pattern, "", stripped_text, flags=re.IGNORECASE)
    stripped_text = re.sub(r"[^\w\s]", "", stripped_text).strip()

    return len(stripped_text) < 4


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
