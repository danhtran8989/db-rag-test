from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings

# ─── PostgreSQL (Supabase compatible) ──────────────────────────────────────
# retrievers.py  (replace only the PostgreSQL function)

def get_postgres_retriever(
    connection_string: str,
    collection_name: str = "chunks",   # table name – keep param name for compatibility
    embedding: Embeddings = None,
    k: int = 4,
) -> BaseRetriever:
    """
    PostgreSQL + pgvector using current langchain-postgres API (create_sync + init table)
    """
    try:
        from langchain_postgres import PGEngine, PGVectorStore
    except ImportError:
        raise ImportError(
            "Please run: pip install --upgrade langchain-postgres psycopg"
        )

    if embedding is None:
        raise ValueError("Embedding model required")

    cleaned = (connection_string or "").strip()
    if not cleaned:
        raise ValueError("Connection string is empty or None")

    # Create engine from connection string
    engine = PGEngine.from_connection_string(url=cleaned)

    # IMPORTANT: Initialize the table if it doesn't exist yet
    # Dimension = 384 for BAAI/bge-small-en-v1.5
    # Run this only once (or when changing dim) – comment out after first success
    try:
        engine.init_vectorstore_table(
            table_name=collection_name,
            vector_size=384,   # ← must match your embeddings dimension!
            # schema_name="public",  # optional – uncomment if using custom schema
        )
    except Exception as init_err:
        # Ignore if table already exists (common error)
        if "already exists" not in str(init_err).lower():
            raise init_err

    # Create the vector store using create_sync (sync version for simplicity)
    vector_store = PGVectorStore.create_sync(
        engine=engine,
        embedding_service=embedding,   # ← note: embedding_service, not embeddings
        table_name=collection_name,
        # schema_name="public",       # optional
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    
# ─── MySQL (unchanged) ─────────────────────────────────────────────────────
def get_mysql_retriever(
    host: str = "localhost",
    port: int = 3306,
    user: str = "root",
    password: str = "",
    database: str = "rag_db",
    table_name: str = "chunks",
    embedding: Embeddings = None,
    k: int = 4,
) -> BaseRetriever:
    try:
        import pymysql
    except ImportError:
        raise ImportError("pip install pymysql")

    if embedding is None:
        raise ValueError("Embedding model required")

    class MySQLVectorRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str) -> List[Document]:
            query_vec = embedding.embed_query(query)
            vec_str = f"[{','.join(map(str, query_vec))}]"

            conn = pymysql.connect(
                host=host, port=port, user=user, password=password,
                database=database, cursorclass=pymysql.cursors.DictCursor
            )
            try:
                with conn.cursor() as cur:
                    sql = f"""
                    SELECT content, metadata,
                           COSINE_SIMILARITY(embedding, VECTOR_FROM_TEXT(%s)) AS similarity
                    FROM {table_name}
                    ORDER BY similarity DESC
                    LIMIT %s
                    """
                    cur.execute(sql, (vec_str, k))
                    rows = cur.fetchall()

                    docs = []
                    for row in rows:
                        meta = row.get("metadata") or {}
                        meta["similarity"] = float(row["similarity"]) if row["similarity"] else 0.0
                        meta["db"] = "MySQL"
                        docs.append(Document(
                            page_content=row["content"],
                            metadata=meta
                        ))
                    return docs
            finally:
                conn.close()

    return MySQLVectorRetriever()

# ─── Oracle & MongoDB (unchanged – can keep as is) ─────────────────────────
# ... (your existing Oracle and MongoDB code here if needed)


