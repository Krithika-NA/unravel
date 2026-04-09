import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = None

def ingest_pdf(pdf_file):
    global vectorstore
    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return f"Ingested {len(chunks)} chunks from PDF"

def query_rag(question):
    if vectorstore is None:
        return "Please upload a PDF first."
    
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    sources = set(doc.metadata.get("page", "?") for doc in docs)
    return f"{answer}\n\n📄 Sources: Pages {sources}"

with gr.Blocks(title="RationAI") as app:
    gr.Markdown("# Unravel — Ask Your Documents Anything")
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        ingest_btn = gr.Button("Ingest PDF")
    ingest_output = gr.Textbox(label="Status")
    ingest_btn.click(ingest_pdf, inputs=pdf_input, outputs=ingest_output)

    gr.Markdown("### Ask a Question")
    question_input = gr.Textbox(label="Your Question")
    ask_btn = gr.Button("Ask")
    answer_output = gr.Textbox(label="Answer", lines=6)
    ask_btn.click(query_rag, inputs=question_input, outputs=answer_output)

app.launch()