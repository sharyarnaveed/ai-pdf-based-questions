import os
def load_chunks_from_file(filename):
    """Load chunks from a text file"""
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = content.split("\n---CHUNK_SEPARATOR---\n")
    return [chunk.strip() for chunk in chunks if chunk.strip()]
