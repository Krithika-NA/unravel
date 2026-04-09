# Unravel — Intelligent Document Query System

Ask anything from any PDF and get accurate, cited answers instantly.

---

## Overview

Unravel is a Retrieval Augmented Generation (RAG) based document intelligence system. It allows users to upload PDF documents, ask questions in natural language, and receive accurate, context-grounded answers with page citations.

Unlike traditional keyword-based search, Unravel understands semantic meaning and retrieves the most relevant information from documents before generating responses.

---

## Features

- Works with any PDF document
- Natural language querying
- Context-aware, grounded answers
- Page-level citations for transparency
- Local embeddings to reduce API cost
- Fast inference using Groq LPU

---

## Tech Stack

| Component | Technology |
|----------|-----------|
| RAG Orchestration | LangChain LCEL |
| Vector Database | ChromaDB |
| Embedding Model | all-MiniLM-L6-v2 (HuggingFace, local) |
| LLM | Llama 3.1 via Groq API |
| Frontend | Gradio |
| Language | Python 3.11 |

---

## Architecture

1. PDF is uploaded and parsed using PyPDF  
2. Text is split into chunks  
3. Chunks are converted into embeddings using a local model  
4. Embeddings are stored in ChromaDB  
5. User query is embedded and relevant chunks are retrieved  
6. Retrieved context is passed to the LLM  
7. LLM generates an answer strictly based on retrieved context  
8. Response is returned with page citations  

---

## Why RAG?

Large Language Models do not have access to private documents and retraining them is expensive.

RAG (Retrieval Augmented Generation) solves this by:

- Retrieving relevant document chunks  
- Providing them as context to the model  
- Generating answers grounded in that context  

This ensures accuracy, reduces hallucination, and enables document-specific querying.

---

## Conclusion

Unravel demonstrates how modern AI systems can be combined with efficient retrieval mechanisms to build practical, real-world applications. By leveraging RAG architecture, it enables accurate and reliable interaction with unstructured data such as PDFs.

The system is designed to be scalable, cost-efficient, and adaptable to multiple domains, making it a strong foundation for future enhancements such as multi-document querying, API-based deployment, and advanced user interfaces.

---

