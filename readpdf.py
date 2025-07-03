import pdfplumber

def extract_text(pdf_path="book.pdf"):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Total pages in PDF: {len(pdf.pages)}")
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:  
                    text += page_text + "\n"
                    print(f"Processed page {page_num}")
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []
    
   
    chunks = []
    sentences = text.split('.')
    
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            sentence_with_period = sentence + ". "
            if len(current_chunk) + len(sentence_with_period) <= 500:
                current_chunk += sentence_with_period
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence_with_period
    

    if current_chunk:
        chunks.append(current_chunk.strip())
    
    print(f"Total chunks created: {len(chunks)}")  # Debug: check chunk count
    return chunks