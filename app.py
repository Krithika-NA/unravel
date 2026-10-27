import os
import gradio as gr
import spaces
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.caches import BaseCache  # noqa: F401 - needed so pydantic can resolve ChatGroq's forward refs
from langchain_core.callbacks import Callbacks  # noqa: F401 - same reason
ChatGroq.model_rebuild()
from langchain_chroma import Chroma
import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re
import uuid
import traceback

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file or environment.")

MAX_FILE_SIZE_MB = 25
NO_ANSWER_MSG = "The document doesn't answer this question."

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=GROQ_API_KEY)


@spaces.GPU
def _zerogpu_startup_check():
    # Unused — this Space is on ZeroGPU hardware, which requires at least one
    # @spaces.GPU-decorated function to pass its startup check. Everything in
    # this app (FastEmbed, Groq API calls) actually runs on CPU; this function
    # is never called, so no GPU time is ever consumed.
    pass


def parse_model_json(text):
    """Parse the model's JSON reply, tolerating code fences and a missing outer brace."""
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    candidates = []
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    candidates.append("{" + text.strip().strip("{}").strip() + "}")
    for c in candidates:
        try:
            parsed = json.loads(c)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


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

        # Each ingest gets its own Chroma database in its own folder under /tmp,
        # so sessions are fully isolated and no client or tenant state is shared.
        collection_name = f"session-{uuid.uuid4().hex}"
        chroma_client = chromadb.PersistentClient(path=f"/tmp/chroma_{collection_name}")

        # Embed in small batches instead of all at once — embedding every chunk in a
        # single call spikes memory and can exceed free-tier memory limits on larger
        # PDFs, causing a silent OOM restart.
        BATCH_SIZE = 10
        vectorstore = None
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    batch, embeddings, client=chroma_client, collection_name=collection_name
                )
            else:
                vectorstore.add_documents(batch)

        session_state["vectorstore"] = vectorstore
        return f"Ingested {len(chunks)} chunks from PDF.", session_state
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        return f"Failed to process PDF: {e}", session_state


def query_rag(question, session_state):
    vectorstore = session_state.get("vectorstore") if session_state else None
    if vectorstore is None:
        return "Please upload and ingest a PDF first."
    if not question or not question.strip():
        return "Please type a question."

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
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
its page number. The context must directly answer the exact question asked.
If it only covers a related but different topic, say the retrieved text does not
answer the question and give an empty used_pages list. Do not guess.
If the context lists several categories, amounts or conditions that answer the question, include all of them. Leave out details that do not answer it.
Respond with ONLY a JSON object, no other text, in this exact shape:
{{"answer": "<your answer>", "used_pages": [<page numbers you actually drew the answer from>]}}
If the context does not answer the question, set "answer" to "The document doesn't answer this question." and use "used_pages": [].
Context:
{context}
Question: {question}
""")

        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({"context": context, "question": question})

        parsed = parse_model_json(raw)

        if parsed is not None:
            answer = str(parsed.get("answer", "")).strip()
            used_pages_raw = parsed.get("used_pages", [])
            if not isinstance(used_pages_raw, list):
                used_pages_raw = []
            used_pages = [p for p in used_pages_raw if isinstance(p, int)]
            if not answer:
                return NO_ANSWER_MSG
            if used_pages:
                return f"{answer}\n\n📄 Sources: Pages {sorted(set(used_pages))}"
            return answer

        # Fallback if the reply couldn't be parsed at all — show the text, and be
        # honest that these pages are only what was retrieved, not confirmed sources.
        all_pages = sorted(set(
            (doc.metadata.get("page", 0) + 1) if isinstance(doc.metadata.get("page"), int) else "?"
            for doc in docs
        ), key=str)
        return f"{raw.strip()}\n\n📄 Retrieved from: Pages {all_pages} (unconfirmed — model didn't return structured citations)"
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        return f"Something went wrong answering that question: {e}"


with gr.Blocks(title="Unravel") as app:
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
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        show_api=False,
    )
