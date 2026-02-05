# app.py
import gradio as gr
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ─── Import retriever factories ────────────────────────────────────────────
from retrievers import (
    get_mysql_retriever,
    get_postgres_retriever,
    get_oracle_retriever,
    get_mongodb_retriever,
)

# ─── Global state ──────────────────────────────────────────────────────────
current_retriever = None

# Lightweight open-source embedding model (replacement for OpenAI's text-embedding-3-small)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Qwen3-1.7B as lightweight open-source LLM
model_id = "Qwen/Qwen3-1.7B"          # or "Qwen/Qwen3-1.7B-Instruct" if you find instruct-tuned variant

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True,
)

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.9,
    repetition_penalty=1.05,
    do_sample=True,
    device_map="auto",
)

llm = ChatHuggingFace(
    llm=HuggingFacePipeline(pipeline=hf_pipeline),
)

# ─── RAG logic ─────────────────────────────────────────────────────────────
def retrieve_and_generate(query: str, history: List):
    global current_retriever

    if current_retriever is None:
        return history + [(query, "Please configure database connection first.")], ""

    try:
        docs: List[Document] = current_retriever.invoke(query)
    except Exception as e:
        return history + [(query, f"Retrieval error: {str(e)}")], ""

    if not docs:
        return history + [(query, "No relevant documents found.")], ""

    context_lines = []
    for i, doc in enumerate(docs, 1):
        score = doc.metadata.get("similarity", doc.metadata.get("score", "n/a"))
        source = doc.metadata.get("source", doc.metadata.get("db", "unknown"))
        context_lines.append(f"[Doc {i} | score={score:.3f} | {source}]\n{doc.page_content.strip()}\n")

    context = "\n".join(context_lines)

    system_prompt = """You are a helpful assistant that answers questions using **only** the provided context.
If the context does not contain enough information, clearly say so and do not make up facts."""

    user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        response = llm.invoke(messages)
        answer = response.content.strip()

    except Exception as e:
        answer = f"LLM error: {str(e)}"

    return history + [(query, answer)], ""


# ─── Database connection setup ─────────────────────────────────────────────
def connect_to_db(db_type: str, **kwargs):
    global current_retriever

    try:
        if db_type == "MySQL":
            current_retriever = get_mysql_retriever(
                host=kwargs.get("host", "localhost"),
                user=kwargs.get("user", "root"),
                password=kwargs.get("password", ""),
                database=kwargs.get("database", "rag_db"),
                port=int(kwargs.get("port", 3306)),
            )

        elif db_type == "PostgreSQL":
            current_retriever = get_postgres_retriever(
                connection_string=kwargs.get("connection_string", "")
            )

        elif db_type == "Oracle":
            current_retriever = get_oracle_retriever(
                dsn=kwargs.get("dsn", ""),
                user=kwargs.get("user", ""),
                password=kwargs.get("password", ""),
            )

        elif db_type == "MongoDB":
            current_retriever = get_mongodb_retriever(
                uri=kwargs.get("uri", ""),
                db_name=kwargs.get("db_name", "rag"),
                collection=kwargs.get("collection", "chunks"),
                index_name=kwargs.get("index_name", "vector_index"),
            )

        else:
            return "Unsupported database type.", None

        return f"Connected to {db_type} successfully!", f"Active: {db_type}"

    except Exception as e:
        current_retriever = None
        return f"Connection failed: {str(e)}", "Not connected"


# ─── Gradio Interface ──────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# RAG Chat – Connect to Your Database (Open-source models)")

    with gr.Tab("Database Connection"):
        db_type = gr.Dropdown(
            choices=["MySQL", "PostgreSQL", "Oracle", "MongoDB"],
            label="Database Type",
            value="PostgreSQL"
        )

        # MySQL fields
        with gr.Group(visible=True) as mysql_group:
            mysql_host = gr.Textbox(label="Host", value="localhost")
            mysql_port = gr.Number(label="Port", value=3306, precision=0)
            mysql_user = gr.Textbox(label="User", value="root")
            mysql_password = gr.Textbox(label="Password", type="password")
            mysql_db = gr.Textbox(label="Database", value="rag_db")

        # PostgreSQL
        with gr.Group(visible=False) as pg_group:
            pg_conn_str = gr.Textbox(
                label="Connection String",
                placeholder="postgresql+psycopg://user:pass@host:5432/dbname",
                lines=2
            )

        # Oracle
        with gr.Group(visible=False) as oracle_group:
            oracle_dsn = gr.Textbox(label="DSN", placeholder="host:port/service_name")
            oracle_user = gr.Textbox(label="User")
            oracle_password = gr.Textbox(label="Password", type="password")

        # MongoDB
        with gr.Group(visible=False) as mongo_group:
            mongo_uri = gr.Textbox(label="Connection URI", placeholder="mongodb+srv://...")
            mongo_db = gr.Textbox(label="Database Name", value="rag")
            mongo_coll = gr.Textbox(label="Collection Name", value="chunks")
            mongo_index = gr.Textbox(label="Vector Index Name", value="vector_index")

        status = gr.Markdown("**Status:** Not connected")
        connect_btn = gr.Button("Connect")

        # Show/hide correct fields
        def update_visibility(db):
            return (
                gr.update(visible=db == "MySQL"),
                gr.update(visible=db == "PostgreSQL"),
                gr.update(visible=db == "Oracle"),
                gr.update(visible=db == "MongoDB"),
            )

        db_type.change(
            update_visibility,
            inputs=db_type,
            outputs=[mysql_group, pg_group, oracle_group, mongo_group]
        )

        # Connect button logic
        connect_inputs = [
            db_type,
            mysql_host, mysql_port, mysql_user, mysql_password, mysql_db,
            pg_conn_str,
            oracle_dsn, oracle_user, oracle_password,
            mongo_uri, mongo_db, mongo_coll, mongo_index,
        ]

        connect_btn.click(
            connect_to_db,
            inputs=connect_inputs,
            outputs=[gr.Textbox(label="Connection Result"), status]
        )

    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(height=580)
        msg = gr.Textbox(placeholder="Ask a question (after connecting to DB) ...", lines=2)
        with gr.Row():
            submit = gr.Button("Send")
            clear = gr.Button("Clear")

        submit.click(
            retrieve_and_generate,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        msg.submit(
            retrieve_and_generate,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        clear.click(lambda: None, None, chatbot)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Chat – Connect to Your Database (Open-source models)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a publicly shareable Gradio link (temporary public URL)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio server on (default: 7860)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    # You can easily add more flags later, e.g.:
    # parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        # debug=args.debug,   # if you add --debug later
    )