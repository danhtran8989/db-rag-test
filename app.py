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

# Lightweight embedding model (384 dim)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Model – using a small, good instruct model suitable for RAG
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True,
)

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.25,
    top_p=0.92,
    repetition_penalty=1.05,
    do_sample=True,
)

llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=hf_pipeline))

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
            {"role": "user", "content": user_prompt},
        ]
        response = llm.invoke(messages)
        answer = response.content.strip()
    except Exception as e:
        answer = f"LLM error: {str(e)}"

    return history + [(query, answer)], ""


# ─── Database connection setup ─────────────────────────────────────────────
def connect_to_db(
    db_type: str,
    mysql_host: str,
    mysql_port: float,
    mysql_user: str,
    mysql_password: str,
    mysql_db: str,
    pg_conn_str: str,
    oracle_dsn: str,
    oracle_user: str,
    oracle_password: str,
    mongo_uri: str,
    mongo_db_name: str,
    mongo_collection: str,
    mongo_index_name: str,
) -> tuple[str, str]:
    global current_retriever

    try:
        if db_type == "MySQL":
            current_retriever = get_mysql_retriever(
                host=mysql_host,
                port=int(mysql_port),
                user=mysql_user,
                password=mysql_password,
                database=mysql_db,
                embedding=embeddings,
            )

        elif db_type == "PostgreSQL":
            if pg_conn_str is None or not isinstance(pg_conn_str, str) or not pg_conn_str.strip():
                return (
                    "Error: Please enter a valid PostgreSQL connection string.",
                    "**Status:** Not connected"
                )
            current_retriever = get_postgres_retriever(
                connection_string=pg_conn_str.strip(),
                embedding=embeddings,
            )

        elif db_type == "Oracle":
            if not oracle_dsn.strip() or not oracle_user.strip():
                return "Error: Oracle DSN and user are required.", "**Status:** Not connected"
            current_retriever = get_oracle_retriever(
                dsn=oracle_dsn.strip(),
                user=oracle_user.strip(),
                password=oracle_password,
                embedding=embeddings,
            )

        elif db_type == "MongoDB":
            if not mongo_uri.strip():
                return "Error: MongoDB URI is required.", "**Status:** Not connected"
            current_retriever = get_mongodb_retriever(
                uri=mongo_uri.strip(),
                db_name=mongo_db_name.strip() or "rag",
                collection=mongo_collection.strip() or "chunks",
                index_name=mongo_index_name.strip() or "vector_index",
                embedding=embeddings,
            )

        else:
            return "Unsupported database type.", "**Status:** Not connected"

        return f"Connected to {db_type} successfully!", f"**Active:** {db_type}"

    except Exception as e:
        current_retriever = None
        return f"Connection failed: {str(e)}", "**Status:** Not connected"


# ─── Gradio Interface ──────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG Chat – Connect to Your Database (Open-source models)")

    with gr.Tab("Database Connection"):
        db_type = gr.Dropdown(
            choices=["MySQL", "PostgreSQL", "Oracle", "MongoDB"],
            label="Database Type",
            value="PostgreSQL"
        )

        with gr.Group(visible=True) as mysql_group:
            mysql_host = gr.Textbox(label="Host", value="localhost")
            mysql_port = gr.Number(label="Port", value=3306, precision=0)
            mysql_user = gr.Textbox(label="User", value="root")
            mysql_password = gr.Textbox(label="Password", type="password")
            mysql_db = gr.Textbox(label="Database", value="rag_db")

        with gr.Group(visible=False) as pg_group:
            pg_conn_str = gr.Textbox(
                label="Connection String",
                placeholder="postgresql+psycopg://user:password@localhost:5432/rag_db",
                lines=2
            )

        with gr.Group(visible=False) as oracle_group:
            oracle_dsn = gr.Textbox(label="DSN", placeholder="localhost:1521/orclpdb1")
            oracle_user = gr.Textbox(label="User")
            oracle_password = gr.Textbox(label="Password", type="password")

        with gr.Group(visible=False) as mongo_group:
            mongo_uri = gr.Textbox(label="Connection URI", placeholder="mongodb://localhost:27017")
            mongo_db_name = gr.Textbox(label="Database Name", value="rag")
            mongo_collection = gr.Textbox(label="Collection Name", value="chunks")
            mongo_index_name = gr.Textbox(label="Vector Index Name", value="vector_index")

        status = gr.Markdown("**Status:** Not connected")
        result_box = gr.Textbox(label="Connection Result", interactive=False)

        connect_btn = gr.Button("Connect")

        # ─── Visibility logic ───────────────────────────────────────
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

        # Connect button – inputs order MUST match function parameters
        connect_btn.click(
            fn=connect_to_db,
            inputs=[
                db_type,
                mysql_host, mysql_port, mysql_user, mysql_password, mysql_db,
                pg_conn_str,
                oracle_dsn, oracle_user, oracle_password,
                mongo_uri, mongo_db_name, mongo_collection, mongo_index_name,
            ],
            outputs=[result_box, status]
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
    parser = argparse.ArgumentParser(description="RAG Chat – Database Vector Search")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
