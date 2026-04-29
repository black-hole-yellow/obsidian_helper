import tiktoken
from config import MAX_CHUNK_TOKENS, CHUNK_OVERLAP

def get_tokenizer():
    # 'cl100k_base' is standard for OpenAI, but serves as a highly accurate 
    # and lightning-fast proxy tokenizer for models like Qwen and Llama.
    return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Returns the exact token count of a string."""
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))

def chunk_text(text: str, max_tokens: int = MAX_CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Intelligently splits text into chunks. 
    If the text is smaller than max_tokens, it returns the whole text.
    """
    if not text.strip():
        return []

    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)

    # ── DYNAMIC SHORT-CIRCUIT ───────────────────────────────────────────────
    # If the entire document fits within our token limit, DO NOT CHUNK IT.
    # This is crucial for small articles to give the LLM full context.
    if total_tokens <= max_tokens:
        print(f"📄 Document is {total_tokens} tokens. Processing as a single chunk.")
        return [text]

    print(f"✂️ Document is {total_tokens} tokens. Splitting into chunks...")
    
    chunks = []
    start_idx = 0

    # ── PRECISION CHUNKING WITH OVERLAP ─────────────────────────────────────
    while start_idx < total_tokens:
        # Get the slice of tokens
        end_idx = min(start_idx + max_tokens, total_tokens)
        chunk_tokens = tokens[start_idx:end_idx]
        
        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move start index forward, subtracting the overlap
        start_idx += (max_tokens - overlap)

    return chunks