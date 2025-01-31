# Ubuntu Documentation Chatbot 🤖📚

**Docker Image**: [Ubuntu RAG Bot](https://hub.docker.com/r/klean05/ubuntu-rag-bot)  
**GitHub Repository**: [KLean05/ubuntu-rag-chatbot](https://github.com/KLean05/ubuntu-rag-chatbot)  

An AI-powered chatbot designed for Ubuntu technical documentation, leveraging **semantic search** and **local LLM processing**. This solution is secure, efficient, and fully deployable on both CPU and NVIDIA GPUs.

---

## ✨ Features  

- **🔍 Semantic Search**: Retrieves relevant answers using FAISS vector store and Sentence Transformers.  
- **🤖 Local LLM Processing**: Powered by **Mistral 7B** model via **Ollama**, ensuring no reliance on cloud services.  
- **🚳 Dockerized**: Simplified deployment with both CPU and GPU support.  
- **📂 Source Citation**: Includes document references in the chatbot's responses.  
- **🔡 Secure and Local**: Operates entirely on your local machine for enhanced privacy.  

---

## 🚀 Quick Start  

### **Using Docker**  

#### Pull and Run the Image  
```bash
# Pull the Docker image
docker pull klean05/ubuntu-rag-bot:latest  

# For CPU deployment
docker run -p 8000:8000 -p 11434:11434 klean05/ubuntu-rag-bot  

# For GPU deployment (requires NVIDIA GPU)
docker run --gpus all -p 8000:8000 -p 11434:11434 klean05/ubuntu-rag-bot  
```

#### Access the API  
Visit **[http://localhost:8000/docs](http://localhost:8000/docs)** to view and interact with the Swagger API documentation.  

---

### **Local Setup**  

#### Clone the Repository  
```bash
git clone https://github.com/KLean05/ubuntu-rag-chatbot  
cd ubuntu-rag-chatbot  
```

#### Install Dependencies  
```bash
pip install -r requirements.txt  
```

#### Start Services  
```bash
ollama serve & python app.py  
```

Access the API at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## 🕠️ Technical Implementation  

The chatbot follows the **Retrieval-Augmented Generation (RAG)** architecture:  

```mermaid
graph TD
    A[Markdown Files] --> B(Chunking)
    B --> C[Embeddings]
    C --> D[FAISS Store]
    D --> E[Query Processing]
    E --> F[LLM (Mistral Model)]
    F --> G[Answer + Sources]
```

### **Chunking Strategy**  

| **Parameter**  | **Value**          | **Rationale**               |  
|-----------------|--------------------|-----------------------------|  
| Chunk Size      | 1000 characters    | Captures full technical concepts. |  
| Overlap         | 200 characters     | Ensures continuity of context.   |  
| Split Points    | Headings, Paragraphs | Preserves document structure.    |  

### **Document Parsing Optimization**
1. Efficient Loading:
Using the DirectoryLoader from langchain_community with multithreading and autodetection of file encoding, which is one of the best approaches when dealing with many documents.

2. Advanced Tokenization:
For even better parsing, we can use tokenizers from libraries like Hugging Face’s transformers that are optimized for speed and can work with pre-trained models.
---

## 📄 API Documentation  

### **Endpoint**: `POST /ask`  

#### **Request Example**  
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain Brand Store setup"}'
```

#### **Response Example**  
```json
{
  "question": "Explain Brand Store setup",
  "answer": "To set up a Brand Store, follow these steps:...",
  "sources": ["brand-store.md"]
}
```

---

## 🗂 Project Structure  

```plaintext
.
├── app.py                 # Main FastAPI application  
├── Dockerfile             # Docker configuration  
├── requirements.txt       # Python dependencies  
├── demo_bot_data/         # Directory for markdown documentation  
│   └── ubuntu-docs/       # Ubuntu documentation files  
├── images                 # screenshots 
├── OUTCOMES.md            # Screenshots and results documentation  
└── README.md              # This file  
```

---

## 🌟 Future Improvements  

- **Hybrid Search**: Combine semantic search with keyword-based search for more robust retrieval.  
- **Enhanced Chunking**: Utilize advanced context-aware chunking strategies for better semantic splits.  

---

## 🎥 Demo Video
Watch the working demo of the Ubuntu Documentation Chatbot in action: ![demo](Demo.mp4)

The demo showcases:

Application startup via Docker.
Using the Swagger UI to interact with the chatbot.
A sample query and its detailed response with cited sources.

---
