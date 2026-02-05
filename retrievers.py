# retrievers.py
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
import json
# ─── PostgreSQL (Supabase compatible) ──────────────────────────────────────
# retrievers.py (replace only the PostgreSQL function)
def get_postgres_retriever(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    collection_name: str = "chunks", # table name – keep param name for compatibility
    embedding: Embeddings = None,
    k: int = 4,
) -> BaseRetriever:
    """
    PostgreSQL + pgvector using direct psycopg2 (like MySQL)
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        raise ImportError(
            "Please run: pip install psycopg2-binary"
        )
    if embedding is None:
        raise ValueError("Embedding model required")
    if not host.strip():
        raise ValueError("Host is empty or None")
    # Initialize table if not exists
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        sslmode="require",
    )
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {collection_name} (
                id SERIAL PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding VECTOR(384)
            );
            """)
            # Optional: cur.execute(f"CREATE INDEX IF NOT EXISTS {collection_name}_hnsw_idx ON {collection_name} USING hnsw (embedding vector_cosine_ops);")
        conn.commit()
    except Exception as init_err:
        # Ignore if table already exists (common error)
        if "already exists" not in str(init_err).lower():
            raise init_err
    finally:
        conn.close()
    class PostgresVectorRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str) -> List[Document]:
            query_vec = embedding.embed_query(query)
            vec_str = '[' + ','.join(map(str, query_vec)) + ']'
            conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                sslmode="require",
                cursor_factory=RealDictCursor
            )
            try:
                with conn.cursor() as cur:
                    sql = f"""
                    SELECT content, metadata,
                           (embedding <=> %s::vector) AS distance
                    FROM {collection_name}
                    ORDER BY distance ASC
                    LIMIT %s
                    """
                    cur.execute(sql, (vec_str, k))
                    rows = cur.fetchall()
                    docs = []
                    for row in rows:
                        meta = json.loads(row["metadata"]) if row["metadata"] else {}
                        meta["similarity"] = 1 - float(row["distance"]) if row["distance"] else 0.0
                        meta["db"] = "PostgreSQL"
                        docs.append(Document(
                            page_content=row["content"],
                            metadata=meta
                        ))
                    return docs
            finally:
                conn.close()
    return PostgresVectorRetriever()
   
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



