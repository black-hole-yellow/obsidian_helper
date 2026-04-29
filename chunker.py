import tiktoken
from config import cfg

# Define constants (you can also move these to config.yaml)
MAX_CHUNK_TOKENS = 1200 
CHUNK_OVERLAP = 100

def chunk(text: str, source_title: str = "") -> list[dict]:
    """
    Splits text into chunks based on token counts.
    If the text is small, it processes as a single chunk.
    """
    if not text.strip():
        return []

    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)

    # DYNAMIC SHORT-CIRCUIT: Don't chunk if it fits in one go
    if total_tokens <= MAX_CHUNK_TOKENS:
        return [{
            "text": text,
            "heading": "Full Content",
            "source_title": source_title,
            "chunk_index": 1,
            "total_chunks": 1
        }]

    chunks = []
    start_idx = 0
    chunk_num = 1
    
    while start_idx < total_tokens:
        end_idx = min(start_idx + MAX_CHUNK_TOKENS, total_tokens)
        chunk_tokens = tokens[start_idx:end_idx]
        
        chunks.append({
            "text": tokenizer.decode(chunk_tokens),
            "heading": f"Part {chunk_num}",
            "source_title": source_title,
            "chunk_index": chunk_num,
            "total_chunks": 0 # Will update below
        })
        
        start_idx += (MAX_CHUNK_TOKENS - CHUNK_OVERLAP)
        chunk_num += 1

    # Update total count
    for c in chunks:
        c["total_chunks"] = len(chunks)
        
    return chunks