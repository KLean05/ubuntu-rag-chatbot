# app.py
import os
import logging
import shutil
from fastapi import FastAPI, HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ubuntu Documentation Chatbot")

# Configuration
DATA_PATH = "../ubuntu-docs"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "mistral"  # 4-bit quantized via Ollama

# --------------------------
# Vector Store Configuration
# --------------------------
# We use FAISS for its efficient similarity search capabilities
# and native integration with HuggingFace embeddings. The index
# is persisted locally to avoid recomputing embeddings between sessions.
def create_vector_store():
    """Create and save FAISS vector store from markdown docs"""
    try:
        logger.info("Loading markdown documents...")
        
        loader = DirectoryLoader(
            DATA_PATH,
            glob="**/*.md",
            use_multithreading=True,
            loader_kwargs={'autodetect_encoding': True}
        )
        docs = loader.load()
        
        if not docs:
            raise ValueError("No markdown files found")

        logger.info(f"Loaded {len(docs)} documents")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n## ", "\n\n### ", "\n\n", "\n", " "]
        )
        
        logger.info("Splitting documents...")
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks")

        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        logger.info("Building FAISS index...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local("ubuntu_faiss_index")
        logger.info("Vector store created successfully")
        return vector_store

    except Exception as e:
        logger.error(f"Vector store creation failed: {str(e)}")
        raise


# --------------------------
# Query Processing Pipeline 
# --------------------------
# The retrieval chain combines semantic search with local LLM processing
# using Ollama's quantized Mistral model for efficient local inference
def initialize_qa_chain():
    """Sets up the question-answering system with safety checks"""
    try:
        logger.info("Initializing QA chain...")
        
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        vector_store = FAISS.load_local(
            "ubuntu_faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Updated Ollama initialization
        llm = Ollama(
            model=LLM_MODEL,
            temperature=0.1,
            num_ctx=4096,
            base_url="http://localhost:11434"  # Explicit Ollama server URL
        )

        # Define the prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Answer the question as directly and concisely as possible, based only on the context provided. 
            Do not write "In the context provided" when starting the answer. 
            Structure the answer in complete sentences as a single cohesive paragraph, summarizing the information clearly.

            Context: {context}

            Question: {question}

            Answer:
            """
        )

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
            retriever=vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "lambda_mult": 0.8}
            ),
            return_source_documents=True
        )

    except Exception as e:
        logger.error(f"QA chain initialization failed: {str(e)}")
        raise

# Initialize with cleanup
try:
    qa_chain = initialize_qa_chain()
except (FileNotFoundError, RuntimeError) as e:
    logger.warning(f"Vector store invalid: {str(e)}")
    logger.info("Recreating vector store...")
    if os.path.exists("ubuntu_faiss_index"):
        shutil.rmtree("ubuntu_faiss_index")
    create_vector_store()
    qa_chain = initialize_qa_chain()

@app.post("/ask")
async def ask_question(question: str):
    """Endpoint to handle user questions"""
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Empty question")
            
        logger.info(f"Processing: {question}")
        result = qa_chain.invoke({"query": question})
        
        return {
            "question": question,
            "answer": result["result"],
            "sources": list({
                os.path.basename(doc.metadata["source"])
                for doc in result["source_documents"]
            })
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)  # Changed to localhost
