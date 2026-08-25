import os
import glob
import tempfile
from typing import List
import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """Downloads YouTube audio and normalizes it to 16kHz mono WAV."""
    raw_output_template = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": raw_output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id", "audio")
        raw_wav = os.path.join(DOWNLOAD_DIR, f"{video_id}.wav")

    # Normalize audio to 16kHz single-channel mono
    normalized_wav = convert_to_wav(raw_wav)
    if os.path.exists(raw_wav) and raw_wav != normalized_wav:
        os.remove(raw_wav)
        
    return normalized_wav


def convert_to_wav(input_path: str) -> str:
    """Converts any audio/video stream to 16kHz mono WAV for Whisper optimal ingestion."""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(DOWNLOAD_DIR, f"{base_name}_normalized.wav")
    
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 5, overlap_seconds: int = 2) -> List[str]:
    """
    Slices audio into manageable payloads with a brief overlap to prevent word truncation.
    """
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    overlap_ms = overlap_seconds * 1000
    
    chunks = []
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    
    start = 0
    idx = 0
    while start < len(audio):
        end = min(start + chunk_ms, len(audio))
        chunk = audio[start:end]
        
        chunk_path = os.path.join(DOWNLOAD_DIR, f"{base_name}_chunk_{idx}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
        
        if end == len(audio):
            break
            
        start += chunk_ms - overlap_ms
        idx += 1
        
    return chunks


def cleanup_temp_audio(file_paths: List[str]) -> None:
    """Deletes temporary chunked audio files after transcription completes."""
    for path in file_paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def process_input(source: str) -> List[str]:
    """Ingests YouTube URL or local media and outputs normalized chunked paths."""
    if source.startswith("http://") or source.startswith("https://"):
        print("📥 Processing remote YouTube stream...")
        wav_path = download_youtube_audio(source)
    else:
        print("📁 Processing local media file...")
        wav_path = convert_to_wav(source)

    print("✂️ Slicing audio payload for Whisper ingestion...")
    chunks = chunk_audio(wav_path, chunk_minutes=5)
    print(f"✅ Audio pipeline ready: {len(chunks)} payload chunk(s) generated.")
    return chunks
