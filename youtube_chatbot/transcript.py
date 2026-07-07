from pathlib import Path

import yt_dlp
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from youtube_chatbot.config import DOWNLOADS_DIR, TRANSCRIPTS_DIR, WHISPER_MODEL_PATH


def extract_video_id(url: str) -> str:
    return url.split("v=")[1].split("&")[0]


def get_video_duration(url: str) -> int:
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        return ydl.extract_info(url, download=False)["duration"]


def fetch_transcript(url: str, video_id: str) -> tuple[Path, str]:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.txt"

    if transcript_path.exists():
        print(f"Transcript already exists at {transcript_path}")
        return transcript_path, transcript_path.read_text(encoding="utf-8")

    try:
        print("Trying to find captions using youtube_transcript_api...")
        transcript_list = YouTubeTranscriptApi().list(video_id)
        transcript_obj = transcript_list.find_manually_created_transcript(["en"])

        if not transcript_obj:
            raise RuntimeError("No transcript found")

        transcript = " ".join(snippet.text for snippet in transcript_obj.fetch())
        print("Found YouTube captions.")

    except (TranscriptsDisabled, NoTranscriptFound):
        from faster_whisper import WhisperModel

        print("Generating captions using faster-whisper...")
        whisper_model = WhisperModel(WHISPER_MODEL_PATH)

        out_dir = DOWNLOADS_DIR / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        audio_files = list(out_dir.glob(f"{video_id}.*"))
        if audio_files:
            audio_path = audio_files[0]
            print(f"Using existing audio at {audio_path}")
        else:
            output_template = out_dir / f"{video_id}.%(ext)s"
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": str(output_template),
                "quiet": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            audio_path = next(out_dir.glob(f"{video_id}.*"))
            print(f"Downloaded audio to {audio_path}")

        segments, _ = whisper_model.transcribe(str(audio_path))
        transcript = ""
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            transcript += segment.text + " "

    transcript_path.write_text(transcript, encoding="utf-8")
    return transcript_path, transcript
