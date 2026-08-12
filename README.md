# Unravel — Intelligent Document Query System

Ask anything from any PDF and get accurate, cited answers instantly.

🔗 **Live demo:** https://krithi1357-unravel.hf.space

## Overview

Unravel is a Retrieval Augmented Generation (RAG) based document intelligence system. It allows users to upload PDF documents, ask questions in natural language, and receive accurate, context-grounded answers with page citations.

Unlike traditional keyword-based search, Unravel understands semantic meaning and retrieves the most relevant information from documents before generating responses.

## Features

- Works with any PDF document
- Natural language querying
- Context-aware, grounded answers with page-level citations, sourced from the exact pages the model used — not just the top retrieved chunks
- Local embeddings to reduce API cost
- Fast inference using Groq LPU
- Deployed and live on Hugging Face Spaces

## Tech Stack

| Component | Technology |
|---|---|
| RAG Orchestration | LangChain LCEL |
| Vector Database | ChromaDB (in-memory, session-scoped) |
| Embedding Model | FastEmbed — BAAI/bge-small-en-v1.5 (ONNX, local) |
| LLM | Llama 3.1 via Groq API |
| Frontend | Gradio |
| Deployment | Hugging Face Spaces |
| Language | Python 3.11 |

## Architecture

1. PDF is uploaded and parsed using PyPDF
2. Text is split into chunks
3. Chunks are embedded in small batches using a local ONNX model and stored in a per-session ChromaDB collection
4. User query is embedded and relevant chunks are retrieved
5. Retrieved context, labeled by page number, is passed to the LLM
6. The LLM returns a structured response naming the exact page(s) it used — not just the pages of every chunk retrieved
7. Answer is returned with verified page citations

## Why RAG?

Large Language Models don't have access to private documents, and retraining them is expensive. RAG solves this by retrieving relevant document chunks, providing them as context to the model, and generating answers grounded in that context — improving accuracy and reducing hallucination.

## What's Solid vs. What's Missing

This is deployed and handles real usage safely, but it isn't a fully hardened production system. Being specific about both:

**In place:**
- Live deployment, reachable without running anything locally
- Error handling for bad PDFs, empty extraction, and API failures
- Per-session isolation — concurrent users don't see or corrupt each other's data
- API keys managed via environment secrets, not hardcoded

**Not yet in place:**
- No persistent storage — each session's vector store is in-memory and resets on restart
- No monitoring or alerting if something breaks
- No authentication or per-user rate limiting beyond a file size cap
- No automated tests or CI/CD pipeline — deploys are manual

## Getting Here: Real Deployment Issues

Getting this from a working local prototype to something live surfaced problems beyond the core RAG logic:

- **Memory limits:** the original PyTorch/sentence-transformers embedding stack was too heavy for free-tier hosting; switched to FastEmbed (ONNX-based) for a much lighter footprint
- **Citation accuracy:** early versions cited every retrieved chunk's page rather than the page actually used; redesigned the LLM output to return structured JSON naming the specific source pages, verified against manual test cases
- **Concurrency:** replaced a global vector store with per-session state so multiple users don't interfere with each other's documents
- **Environment-specific bugs:** dependency version mismatches and filesystem/persistence issues that only appeared in the deployed environment, not locally

## Running Locally

```bash
git clone https://github.com/Krithika-NA/unravel.git
cd unravel
pip install -r requirements.txt
# Add GROQ_API_KEY=your_key to a .env file
python app.py
```

## Possible Future Enhancements

- Persistent storage across sessions
- Monitoring and basic usage analytics
- Multi-document querying
