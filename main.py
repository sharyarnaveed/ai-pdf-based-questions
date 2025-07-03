from sentence_transformers import SentenceTransformer
from readpdf import extract_text
import faiss
import numpy as np
import os
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os

load_dotenv() 


model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract chunks with error handling
chunks = extract_text()
if not chunks:
    print("❌ No chunks extracted from PDF!")
    exit()
    
# Create embeddings
embeddings = model.encode(chunks)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings.astype('float32'))

# Search query
print("Question: ")
question = input()
question_embedding = model.encode([question]).astype("float32")

top_k = 10  
distances, indices = index.search(question_embedding, top_k)


# Show results with similarity scores
for idx, (distance, chunk_idx) in enumerate(zip(distances[0], indices[0])):
    
    
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)



context= "\n\n".join([chunks[i] for i in indices[0]])

prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

# Ask Gemini
response = llm.invoke(prompt)

print("\n🧠 Gemini Answer:")
print(response.content)