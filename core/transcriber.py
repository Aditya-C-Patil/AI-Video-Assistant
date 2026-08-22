import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def transcribe_to_english(audio_path: str) -> str:
    """
    Takes any audio chunk (Hindi, Hinglish, or English)
    and directly outputs English text using Groq's whisper-large-v3.
    """
    if not client:
        raise RuntimeError("GROQ_API_KEY is not set in environment / .env")

    with open(audio_path, "rb") as file:
        response = client.audio.translations.create(
            file=(Path(audio_path).name, file.read()),
            model="whisper-large-v3",
            response_format="text",
        )
    return response.strip()


def transcribe_all(chunks: list) -> str:
    """Processes all audio chunks and returns a single combined English transcript."""
    full_transcript = []

    for i, chunk in enumerate(chunks):
        print(f"  → Processing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_to_english(chunk)
        if text:
            full_transcript.append(text)

    print("Transcription complete.")
    return " ".join(full_transcript).strip()