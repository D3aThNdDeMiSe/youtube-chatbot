from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

from youtube_chatbot.config import (
    CAG_CACHE_VERSION,
    CAG_SYSTEM_PROMPT,
    KV_CACHES_DIR,
    TRANSCRIPT_LENGTHS_DIR,
)


def _cache_paths(video_id: str) -> tuple[Path, Path, Path]:
    return (
        KV_CACHES_DIR / f"{video_id}.pt",
        TRANSCRIPT_LENGTHS_DIR / f"{video_id}_length.pt",
        TRANSCRIPT_LENGTHS_DIR / f"{video_id}_cache_version.pt",
    )


def build_or_load_cache(
    video_id: str,
    transcript_text: str,
    query_model,
    tokenizer,
    device: torch.device,
) -> int:
    KV_CACHES_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_LENGTHS_DIR.mkdir(parents=True, exist_ok=True)

    cache_path, length_path, version_path = _cache_paths(video_id)
    cache_valid = (
        cache_path.exists()
        and length_path.exists()
        and version_path.exists()
        and torch.load(version_path, weights_only=False) == CAG_CACHE_VERSION
    )

    if cache_valid:
        prefix_length = torch.load(length_path, weights_only=False)
        print(f"Loaded KV cache ({prefix_length} prefix tokens)")
        return prefix_length

    prefill_messages = [
        {
            "role": "system",
            "content": f"{CAG_SYSTEM_PROMPT}\n\nTranscript:\n{transcript_text}",
        }
    ]
    prefill_inputs = tokenizer.apply_chat_template(
        prefill_messages,
        return_tensors="pt",
        add_generation_prompt=False,
    ).to(device)

    prefix_length = prefill_inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = query_model(**prefill_inputs, use_cache=True)

    kv_cache = outputs.past_key_values
    if hasattr(kv_cache, "to_legacy_cache"):
        kv_cache = kv_cache.to_legacy_cache()

    torch.save(kv_cache, cache_path)
    torch.save(prefix_length, length_path)
    torch.save(CAG_CACHE_VERSION, version_path)
    print(f"Built and saved KV cache ({prefix_length} prefix tokens)")
    return prefix_length


def cag_query(
    video_id: str,
    transcript_text: str,
    query_model,
    tokenizer,
    device: torch.device,
    query: str,
    max_new_tokens: int = 256,
) -> str:
    cache_path, length_path, _ = _cache_paths(video_id)
    prefix_length = torch.load(length_path, weights_only=False)

    query_messages = [
        {
            "role": "system",
            "content": f"{CAG_SYSTEM_PROMPT}\n\nTranscript:\n{transcript_text}",
        },
        {"role": "user", "content": query},
    ]
    full_inputs = tokenizer.apply_chat_template(
        query_messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)

    new_input_ids = full_inputs.input_ids[:, prefix_length:]

    kv_cache = torch.load(cache_path, weights_only=False, map_location=device)
    if isinstance(kv_cache, tuple):
        kv_cache = DynamicCache.from_legacy_cache(kv_cache)

    with torch.no_grad():
        output = query_model.generate(
            input_ids=new_input_ids,
            attention_mask=full_inputs.attention_mask,
            past_key_values=kv_cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)
