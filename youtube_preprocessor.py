"""
preprocessors/youtube_preprocessor.py

Handles YouTube URLs and local video/audio files.

Strategy (in order of preference):
  1. Fetch YouTube auto-generated captions (fastest, no GPU needed)
  2. Download audio → transcribe with local Whisper (offline, private)
  3. Accept a pre-provided transcript string (manual fallback)

All paths produce the same output dict format.
"""

import re
import tempfile
from pathlib import Path


# ── Main entry ───────────────────────────────────────────────────────────────

def process(source: str, prefer_captions: bool = True) -> dict:
    """
    Process a YouTube URL or local video/audio path.

    Args:
        source:           YouTube URL or path to audio/video file
        prefer_captions:  If True, try captions first (faster).
                          If False, always use Whisper transcription.

    Returns:
        {
            "text":     str,   # full transcript text
            "sections": list,  # chunked by time or paragraph
            "metadata": dict   # title, channel, duration, source_url
        }
    """
    if _is_youtube_url(source):
        return _process_youtube(source, prefer_captions)
    elif Path(source).exists():
        return _process_local_audio(Path(source))
    else:
        raise ValueError(f"Source is not a valid URL or file path: {source}")


def process_raw_transcript(text: str, title: str = "Video Transcript") -> dict:
    """
    Accept a manually provided transcript string.
    Use this when you already have the text (saves time and compute).
    """
    from text_preprocessor import process as text_process
    result = text_process(text)
    result["metadata"]["detected_title"] = title
    result["metadata"]["source"] = "manual_transcript"
    return result


# ── YouTube caption fetch ────────────────────────────────────────────────────

def _process_youtube(url: str, prefer_captions: bool) -> dict:
    try:
        import yt_dlp
    except ImportError:
        raise ImportError("yt-dlp not installed. Run: pip install yt-dlp")

    meta = _fetch_youtube_metadata(url)

    # Try captions first
    if prefer_captions:
        transcript = _fetch_captions(url)
        if transcript:
            print(f"[youtube] Using auto-captions for: {meta.get('title', url)}")
            return _build_result(transcript, meta, method="captions")

    # Fall back to Whisper
    print(f"[youtube] Captions unavailable — downloading audio for Whisper...")
    return _transcribe_youtube(url, meta)


def _fetch_youtube_metadata(url: str) -> dict:
    """Get video metadata without downloading."""
    try:
        import yt_dlp
        opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "detected_title": info.get("title", "Unknown"),
                "channel":        info.get("uploader", "Unknown"),
                "duration_sec":   info.get("duration", 0),
                "source_url":     url,
            }
    except Exception:
        return {"detected_title": "Unknown", "source_url": url}


def _fetch_captions(url: str) -> str | None:
    """Try to get auto-generated or manual captions via yt-dlp."""
    try:
        import yt_dlp

        opts = {
            "quiet":         True,
            "skip_download": True,
            "writesubtitles":     True,
            "writeautomaticsub":  True,
            "subtitleslangs":     ["en"],
            "subtitlesformat":    "vtt",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            opts["outtmpl"] = f"{tmpdir}/%(id)s.%(ext)s"
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])

            # Find downloaded .vtt file
            vtt_files = list(Path(tmpdir).glob("*.vtt"))
            if not vtt_files:
                return None

            raw_vtt = vtt_files[0].read_text(encoding="utf-8")
            return _parse_vtt(raw_vtt)

    except Exception as e:
        print(f"[youtube] Caption fetch failed: {e}")
        return None


def _transcribe_youtube(url: str, meta: dict) -> dict:
    """Download audio and transcribe with Whisper."""
    try:
        import yt_dlp
        import whisper
    except ImportError:
        raise ImportError(
            "Whisper not installed. Run: pip install openai-whisper"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "audio.mp3"

        opts = {
            "format":    "bestaudio/best",
            "outtmpl":   str(audio_path.with_suffix("")),
            "quiet":     True,
            "postprocessors": [{
                "key":            "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }],
        }

        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        # Find the actual downloaded file
        mp3_files = list(Path(tmpdir).glob("*.mp3"))
        if not mp3_files:
            raise RuntimeError("Audio download failed — no .mp3 found.")

        print("[youtube] Transcribing with Whisper (this may take a few minutes)...")
        model      = whisper.load_model("base")  # "base" runs fast on M3
        result     = model.transcribe(str(mp3_files[0]))
        transcript = result["text"]

    return _build_result(transcript, meta, method="whisper")


# ── Local audio/video file ───────────────────────────────────────────────────

def _process_local_audio(path: Path) -> dict:
    """Transcribe a local audio or video file with Whisper."""
    try:
        import whisper
    except ImportError:
        raise ImportError("Whisper not installed. Run: pip install openai-whisper")

    print(f"[youtube] Transcribing local file: {path.name}")
    model      = whisper.load_model("base")
    result     = model.transcribe(str(path))
    transcript = result["text"]

    meta = {
        "detected_title": path.stem,
        "source_url":     str(path),
    }

    return _build_result(transcript, meta, method="whisper_local")


# ── Output builder ───────────────────────────────────────────────────────────

def _build_result(transcript: str, meta: dict, method: str) -> dict:
    """Pass transcript through text preprocessor for consistent output."""
    from text_preprocessor import process as text_process

    result = text_process(transcript)
    result["metadata"].update(meta)
    result["metadata"]["transcription_method"] = method
    return result


# ── VTT parser ───────────────────────────────────────────────────────────────

def _parse_vtt(vtt: str) -> str:
    """
    Strip WebVTT timestamps and return clean transcript text.
    Deduplicates overlapping caption lines (common in auto-captions).
    """
    lines  = vtt.split("\n")
    seen   = set()
    output = []

    timestamp = re.compile(r"\d{2}:\d{2}:\d{2}")

    for line in lines:
        line = line.strip()
        # Skip headers, timestamps, cue settings
        if not line or line.startswith("WEBVTT") or timestamp.search(line):
            continue
        # Strip HTML tags from captions
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line and line not in seen:
            seen.add(line)
            output.append(line)

    return " ".join(output)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_youtube_url(s: str) -> bool:
    return "youtube.com" in s or "youtu.be" in s
