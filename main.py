from sentence_transformers import SentenceTransformer
from readpdf import extract_text
import faiss
import numpy as np
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from loadchunks import load_chunks_from_file
from saveembeddings import save_embeddings
from loadembeddings import load_embeddings


load_dotenv() 
EMBEDDINGS_FILE = "embeddings.pkl"
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.txt"
model = SentenceTransformer("all-MiniLM-L6-v2")


chunks = load_chunks_from_file(CHUNKS_FILE)
embeddings = load_embeddings(EMBEDDINGS_FILE)


# Extract chunks with error handling
if chunks is None or embeddings is None:
    print("📄 Extracting text from PDF...")
    chunks = extract_text()
    if not chunks:
        print("❌ No chunks extracted from PDF!")
        exit()
    
    print("🔄 Creating embeddings...")
    embeddings = model.encode(chunks)
    
    save_embeddings(embeddings, EMBEDDINGS_FILE)
    print("💾 Chunks and embeddings saved!")
else:
    print("📁 Loading existing chunks and embeddings...")


if os.path.exists(INDEX_FILE):
    print("📂 Loading existing FAISS index...")
    index = faiss.read_index(INDEX_FILE)
else:
    print("🔍 Creating new FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    # Save the index
    faiss.write_index(index, INDEX_FILE)
    print("💾 FAISS index saved!")


# Search query
while True:
    print("\nQuestion: ")
    question = input()
    if question == "-1":
        break
    else:
        question_embedding = model.encode([question]).astype("float32")

        top_k = 10  
        distances, indices = index.search(question_embedding, top_k)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        context = "\n\n".join([chunks[i] for i in indices[0]])

        prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

        # Ask Gemini
        response = llm.invoke(prompt)

        print("\n🧠 Gemini Answer:")
        print(response.content)