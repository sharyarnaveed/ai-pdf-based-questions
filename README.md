# 📄 AI-Powered PDF Q&A Chatbot

Ask questions about any PDF — get accurate, AI-powered answers based on its actual content.

This project uses:
- 🧠 **Sentence Transformers** to convert PDF chunks into semantic vectors
- ⚡ **FAISS** to search those chunks by similarity
- 🤖 **Gemini 1.5 Flash** (Google) to generate answers using the most relevant text

---

## 📌 Features

- Extracts and chunks text from any PDF
- Converts text into embeddings using `all-MiniLM-L6-v2`
- Searches top-matching chunks with FAISS
- Sends relevant context to Gemini to answer questions
- Fast, accurate, and cost-efficient (using free Gemini API)

---

## 🖼️ Project Flow

```mermaid
graph TD
    A[PDF] --> B[Extract Text]
    B --> C[Split into Chunks]
    C --> D[Create Embeddings]
    D --> E[Store in FAISS Vector DB]
    F[User Question] --> G[Convert to Embedding]
    G --> H[Search in FAISS]
    H --> I[Top Matching Chunks]
    I --> J[Send Context + Question to Gemini]
    J --> K[Get AI-Powered Answer]
#
