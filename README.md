# Unravel

Ask questions about a PDF and get answers with page-level citations. Unravel is a retrieval-augmented generation (RAG) app: it splits your document into chunks, finds the ones most relevant to your question, and has an LLM answer using only those chunks. If the document doesn't answer the question, it says so.

Live demo: https://krithi1357-unravel.hf.space

## How it works

1. Upload a PDF and click Ingest PDF.
2. The PDF is split into 1000-character chunks with 200 characters of overlap.
3. Each chunk is embedded with BAAI/bge-small-en-v1.5 (FastEmbed, ONNX, runs on CPU) and stored in ChromaDB.
4. When you ask a question, the 8 most similar chunks are retrieved and labelled with their page numbers.
5. The Groq-hosted model (openai/gpt-oss-20b) answers using only those chunks and returns the page numbers it used.
6. If the retrieved text doesn't answer the question, the app says so instead of guessing.

Each ingest gets its own ChromaDB collection, so one user's document is never mixed with another's.

## Stack

- Python
- LangChain
- ChromaDB (vector store)
- FastEmbed with BAAI/bge-small-en-v1.5 (embeddings)
- Groq API, openai/gpt-oss-20b (LLM)
- Gradio (interface)
- Hugging Face Spaces (hosting)

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

Create a .env file with your Groq API key:

```
GROQ_API_KEY=your_key_here
```

Then start the app:

```bash
python app.py
```

Open http://localhost:7860.

## Limitations

- Text-based PDFs only. Scanned or image-only PDFs aren't supported (there is no OCR).
- Maximum file size is 25 MB.
- The vector store is in the session only. Documents are not saved after the session ends.
- The free Groq tier has per-minute and per-day token limits, so heavy use can return rate-limit errors.

## Future enhancements

- Multi-document support: query across several PDFs at once, with citations that name the file as well as the page.
- Better retrieval: hybrid search (keyword plus vector) and a reranking step, so the right section is found even when the question is worded differently from the document.
- Evaluation set: a fixed list of questions with expected answers and pages, so retrieval and prompt changes can be measured instead of checked by hand.



