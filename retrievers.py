# retrievers.py
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings

# ─── PostgreSQL (using pgvector + langchain-postgres) ───────────────────────
def get_postgres_retriever(
    connection_string: str,
    collection_name: str = "chunks",           # table name
    embedding: Embeddings = None,              # pass from app.py
    k: int = 4,
    score_threshold: Optional[float] = None,   # optional distance cutoff
) -> BaseRetriever:
    """
    PostgreSQL + pgvector retriever.
    Assumes table 'chunks' with columns:
    - id (serial/pk)
    - content (text)
    - embedding (vector(384) or vector(1536) etc.)
    - metadata (jsonb, optional)
    """
    try:
        from langchain_postgres import PGVector
    except ImportError:
        raise ImportError("Please install langchain-postgres: pip install langchain-postgres")

    if embedding is None:
        raise ValueError("Embedding model must be provided for PGVector")

    vector_store = PGVector(
        connection=connection_string,
        collection_name=collection_name,
        embeddings=embedding,
        # distance_strategy="cosine",  # or "l2", "inner_product" — match your index
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold" if score_threshold else "similarity",
        search_kwargs={
            "k": k,
            **({"score_threshold": score_threshold} if score_threshold else {}),
        }
    )

    # Optional: wrap to normalize score to similarity (1 - distance) if needed
    return retriever


# ─── MySQL (using mysql + manual vector search or third-party) ──────────────
# def get_mysql_retriever(
#     host: str = "localhost",
#     user: str = "root",
#     password: str = "",
#     database: str = "rag_db",
#     port: int = 3306,
#     table_name: str = "chunks",
#     embedding: Embeddings = None,
#     k: int = 4,
# ) -> BaseRetriever:
#     """
#     MySQL vector similarity search.
#     MySQL 8.0+ / 8.4+ supports vector type and cosine similarity natively (2025+).
#     Assumes table:
#     CREATE TABLE chunks (
#         id BIGINT AUTO_INCREMENT PRIMARY KEY,
#         content TEXT,
#         embedding VECTOR(384),          -- adjust dim
#         metadata JSON,
#         INDEX idx_embedding USING VECTOR (embedding)
#     );
#     """
#     try:
#         import pymysql
#         from langchain_community.vectorstores import MySQL
#         # Note: official langchain MySQL vectorstore is limited / community-only
#         # Many people use custom retriever or mysql-vector extension
#     except ImportError:
#         raise ImportError("Please install pymysql and/or mysql-connector-python")

#     if embedding is None:
#         raise ValueError("Embedding model required")

#     # Simple custom retriever using raw SQL (recommended until better integration)
#     class MySQLVectorRetriever(BaseRetriever):
#         def _get_relevant_documents(self, query: str) -> List[Document]:
#             query_vec = embedding.embed_query(query)
#             vec_str = f"[{','.join(map(str, query_vec))}]"

#             conn = pymysql.connect(
#                 host=host, user=user, password=password,
#                 database=database, port=port, cursorclass=pymysql.cursors.DictCursor
#             )
#             try:
#                 with conn.cursor() as cur:
#                     # MySQL 8.4+ vector cosine similarity
#                     sql = f"""
#                     SELECT content, metadata,
#                            COSINE_SIMILARITY(embedding, VECTOR_FROM_TEXT('{vec_str}')) AS similarity
#                     FROM {table_name}
#                     ORDER BY similarity DESC
#                     LIMIT %s
#                     """
#                     cur.execute(sql, (k,))
#                     rows = cur.fetchall()

#                     docs = []
#                     for row in rows:
#                         meta = row.get("metadata") or {}
#                         meta["similarity"] = float(row["similarity"]) if row["similarity"] is not None else 0.0
#                         meta["db"] = "MySQL"
#                         docs.append(Document(
#                             page_content=row["content"],
#                             metadata=meta
#                         ))
#                     return docs
#             finally:
#                 conn.close()

#     return MySQLVectorRetriever()


# # ─── Oracle (using langchain-oracledb) ──────────────────────────────────────
# def get_oracle_retriever(
#     dsn: str,
#     user: str,
#     password: str,
#     table_name: str = "chunks",
#     embedding: Embeddings = None,
#     k: int = 4,
# ) -> BaseRetriever:
#     """
#     Oracle Database 23ai+ vector support via langchain-oracledb
#     """
#     try:
#         from langchain_oracledb import OracleVS
#     except ImportError:
#         raise ImportError("pip install langchain-oracledb oracledb")

#     if embedding is None:
#         raise ValueError("Embedding model required")

#     vector_store = OracleVS.from_existing(
#         embedding_function=embedding,
#         collection_name=table_name,
#         dsn=dsn,
#         user=user,
#         password=password,
#     )

#     return vector_store.as_retriever(search_kwargs={"k": k})


# # ─── MongoDB (Atlas Vector Search recommended) ──────────────────────────────
# def get_mongodb_retriever(
#     uri: str,
#     db_name: str = "rag",
#     collection: str = "chunks",
#     index_name: str = "vector_index",
#     embedding: Embeddings = None,
#     k: int = 4,
# ) -> BaseRetriever:
#     """
#     MongoDB Atlas Vector Search via langchain-mongodb
#     Requires Atlas Vector Search index named 'vector_index' on field 'embedding'
#     """
#     try:
#         from langchain_mongodb import MongoDBAtlasVectorSearch
#         from pymongo import MongoClient
#     except ImportError:
#         raise ImportError("pip install langchain-mongodb pymongo")

#     if embedding is None:
#         raise ValueError("Embedding model required")

#     client = MongoClient(uri)
#     collection_obj = client[db_name][collection]

#     vector_store = MongoDBAtlasVectorSearch(
#         collection=collection_obj,
#         embedding=embedding,
#         index_name=index_name,
#         # embedding_key="embedding",   # default
#         # text_key="content",          # adjust if your field is different
#     )

#     return vector_store.as_retriever(search_kwargs={"k": k})

