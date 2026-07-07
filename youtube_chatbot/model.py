import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from youtube_chatbot.config import QUERY_MODEL


def load_query_model() -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    query_model = AutoModelForCausalLM.from_pretrained(
        QUERY_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return query_model, tokenizer, device
