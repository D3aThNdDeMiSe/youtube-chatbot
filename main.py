import argparse

import torch
from dotenv import load_dotenv

from youtube_chatbot.cag import build_or_load_cache, cag_query
from youtube_chatbot.config import RAG_THRESHOLD_SECONDS
from youtube_chatbot.model import load_query_model
from youtube_chatbot.rag import build_chroma_index, rag_query, semantic_chunk
from youtube_chatbot.transcript import extract_video_id, fetch_transcript, get_video_duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions about a YouTube video transcript using CAG or RAG."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--query",
        default="What did Rachel do?",
        help="Question to ask about the video",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    video_id = extract_video_id(args.url)
    audio_length = get_video_duration(args.url)
    transcript_path, transcript = fetch_transcript(args.url, video_id)

    print(f"Video ID: {video_id}")
    print(f"Duration: {audio_length // 60}m {audio_length % 60}s")

    query_model, tokenizer, device = load_query_model()

    if audio_length <= RAG_THRESHOLD_SECONDS:
        print("Using CAG pipeline (video <= 10 minutes)")
        build_or_load_cache(video_id, transcript, query_model, tokenizer, device)
        response = cag_query(
            video_id,
            transcript,
            query_model,
            tokenizer,
            device,
            args.query,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        print("Using RAG pipeline (video > 10 minutes)")
        chunks = semantic_chunk(transcript)
        chroma_client = build_chroma_index(chunks)
        response = rag_query(
            chroma_client,
            query_model,
            tokenizer,
            device,
            args.query,
            max_new_tokens=args.max_new_tokens,
        )

    print("\n--- Response ---")
    print(response)


if __name__ == "__main__":
    main()
