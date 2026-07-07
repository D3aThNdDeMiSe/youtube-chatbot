import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer

from youtube_chatbot.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNKING_MODEL,
    EMBEDDING_MODEL,
)


def semantic_chunk(transcript: str) -> list[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chunking_model = AutoModelForCausalLM.from_pretrained(
        CHUNKING_MODEL,
        torch_dtype="auto",
    )
    chunking_model.to(device)
    chunking_tokenizer = AutoTokenizer.from_pretrained(CHUNKING_MODEL)

    messages = [
        {
            "role": "system",
            "content": (
                "Your job is to act as a semantic chunker.\n"
                "The goal is to separate the content into semantically relevant groupings.\n"
                'Each group must be delimited by the "段" character.'
            ),
        },
        {
            "role": "user",
            "content": f"Segment this text:\n\n```text\n{transcript}\n```",
        },
    ]
    chunking_inputs = chunking_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = chunking_tokenizer([chunking_inputs], return_tensors="pt").to(device)

    generated_ids = chunking_model.generate(**model_inputs, max_new_tokens=2048)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = chunking_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    preprocessed_transcript = response.split("```text")[1]
    chunks = [chunk.strip() for chunk in preprocessed_transcript.split("段") if chunk.strip()]

    del chunking_model, chunking_tokenizer, model_inputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Created {len(chunks)} semantic chunks")
    return chunks


def build_chroma_index(chunks: list[str]) -> Chroma:
    docs = [Document(page_content=chunk) for chunk in chunks]
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    chroma_client = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embedding_function,
        persist_directory=str(CHROMA_DIR),
    )
    chroma_client.add_documents(docs)
    print(f"Indexed {len(docs)} chunks into Chroma")
    return chroma_client


def rag_query(
    chroma_client: Chroma,
    query_model,
    tokenizer,
    device: torch.device,
    query: str,
    max_new_tokens: int = 256,
) -> str:
    retrieved = chroma_client.similarity_search(query, k=3)
    context = "\n\n".join(doc.page_content for doc in retrieved)
    formal_query = f"Respond to the following text: \n {query} \n {context}"

    messages = [
        {"role": "system", "content": "Answer questions based on the provided video transcript."},
        {"role": "user", "content": formal_query},
    ]
    formatted_query = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenized_query = tokenizer(formatted_query, return_tensors="pt").to(device)

    with torch.no_grad():
        generated = query_model.generate(**tokenized_query, max_new_tokens=max_new_tokens)

    return tokenizer.decode(generated[0], skip_special_tokens=True)
