import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.caches import BaseCache  
from langchain_core.callbacks import Callbacks  
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
import re

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file or environment.")

MAX_FILE_SIZE_MB = 20

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

def ingest_pdf(pdf_file, session_state):
    if pdf_file is None:
        return "Please choose a PDF first.", session_state

    size_mb = os.path.getsize(pdf_file.name) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return f"File is {size_mb:.1f}MB — please upload a PDF under {MAX_FILE_SIZE_MB}MB.", session_state

    try:
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()
        if not documents:
            return "Couldn't extract any text from that PDF (it may be scanned/image-only).", session_state

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        # Each session gets its own in-memory Chroma collection — no cross-user leakage
        session_state["vectorstore"] = Chroma.from_documents(chunks, embeddings)
        return f"Ingested {len(chunks)} chunks from PDF.", session_state
    except Exception as e:
        return f"Failed to process PDF: {e}", session_state

def query_rag(question, session_state):
    vectorstore = session_state.get("vectorstore") if session_state else None
    if vectorstore is None:
        return "Please upload and ingest a PDF first."
    if not question or not question.strip():
        return "Please type a question."

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(question)
        if not docs:
            return "No relevant content found in the document for that question."

        # Label each chunk with its human-readable (1-indexed) page number so the
        # LLM can tell us exactly which page(s) it actually drew the answer from.
        labeled_chunks = []
        for doc in docs:
            page_0indexed = doc.metadata.get("page", None)
            page_label = page_0indexed + 1 if isinstance(page_0indexed, int) else "?"
            labeled_chunks.append(f"[Page {page_label}]\n{doc.page_content}")
        context = "\n\n---\n\n".join(labeled_chunks)

        prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below. Each chunk is labeled with
its page number. If the answer is not contained in the context, say you don't
know — do not guess.

Respond with ONLY a JSON object, no other text, in this exact shape:
{{"answer": "<your answer>", "used_pages": [<page numbers you actually drew the answer from>]}}

If you don't know the answer, use "used_pages": [].

Context:
{context}

Question: {question}
""")

        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({"context": context, "question": question})

        # Be defensive: strip markdown code fences if the model adds them anyway
        cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

        try:
            parsed = json.loads(cleaned)
            answer = str(parsed.get("answer", "")).strip()
            used_pages_raw = parsed.get("used_pages", [])
            used_pages = [p for p in used_pages_raw if isinstance(p, int)]
            if not answer:
                raise ValueError("empty answer field")
            if used_pages:
                return f"{answer}\n\n📄 Sources: Pages {sorted(set(used_pages))}"
            return answer
        except (json.JSONDecodeError, ValueError):
            # Fallback if the model didn't return valid JSON — still show something
            # useful rather than erroring out, but be honest these are retrieved
            # chunks, not confirmed sources.
            all_pages = sorted(set(
                (doc.metadata.get("page", 0) + 1) if isinstance(doc.metadata.get("page"), int) else "?"
                for doc in docs
            ))
            return f"{raw.strip()}\n\n📄 Retrieved from: Pages {all_pages} (unconfirmed — model didn't return structured citations)"
    except Exception as e:
        return f"Something went wrong answering that question: {e}"

with gr.Blocks(title="RationAI") as app:
    session_state = gr.State(value={})

    gr.Markdown("# Unravel — Ask Your Documents Anything")
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        ingest_btn = gr.Button("Ingest PDF")
    ingest_output = gr.Textbox(label="Status")
    ingest_btn.click(
        ingest_pdf,
        inputs=[pdf_input, session_state],
        outputs=[ingest_output, session_state],
    )

    gr.Markdown("### Ask a Question")
    question_input = gr.Textbox(label="Your Question")
    ask_btn = gr.Button("Ask")
    answer_output = gr.Textbox(label="Answer", lines=6)
    ask_btn.click(query_rag, inputs=[question_input, session_state], outputs=answer_output)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        show_api=False,
    )
