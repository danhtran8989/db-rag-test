from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings

# ─── PostgreSQL (pgvector) – safe & modern version ─────────────────────────
def get_postgres_retriever(
    connection_string: str,
    collection_name: str = "chunks",
    embedding: Embeddings = None,
    k: int = 4,
) -> BaseRetriever:
    try:
        from langchain_postgres import PGEngine, PGVectorStore
    except ImportError:
        raise ImportError("Please install:  pip install langchain-postgres")

    if embedding is None:
        raise ValueError("Embedding model must be provided for PGVector")

    if connection_string is None:
        raise ValueError("Connection string is None – check Gradio input")
    if not isinstance(connection_string, str):
        raise TypeError(f"Connection string must be str, got {type(connection_string)}")

    cleaned = connection_string.strip()
    if not cleaned:
        raise ValueError("Connection string is empty")

    # Create engine
    engine = PGEngine.from_connection_string(url=cleaned)

    # Create vector store
    vector_store = PGVectorStore(
        engine=engine,
        collection_name=collection_name,
        embeddings=embedding,
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ─── MySQL (custom – works with MySQL 8.4+ vector) ─────────────────────────
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


# ─── Oracle (23ai+) ────────────────────────────────────────────────────────
def get_oracle_retriever(
    dsn: str,
    user: str,
    password: str,
    table_name: str = "chunks",
    embedding: Embeddings = None,
    k: int = 4,
) -> BaseRetriever:
    try:
        from langchain_oracledb import OracleVS
    except ImportError:
        raise ImportError("pip install langchain-oracledb oracledb")

    if embedding is None:
        raise ValueError("Embedding model required")

    vector_store = OracleVS.from_existing(
        embedding_function=embedding,
        collection_name=table_name,
        dsn=dsn,
        user=user,
        password=password,
    )

    return vector_store.as_retriever(search_kwargs={"k": k})


# ─── MongoDB Atlas Vector Search ───────────────────────────────────────────
def get_mongodb_retriever(
    uri: str,
    db_name: str = "rag",
    collection: str = "chunks",
    index_name: str = "vector_index",
    embedding: Embeddings = None,
    k: int = 4,
) -> BaseRetriever:
    try:
        from langchain_mongodb import MongoDBAtlasVectorSearch
        from pymongo import MongoClient
    except ImportError:
        raise ImportError("pip install langchain-mongodb pymongo")

    if embedding is None:
        raise ValueError("Embedding model required")

    client = MongoClient(uri)
    collection_obj = client[db_name][collection]

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection_obj,
        embedding=embedding,
        index_name=index_name,
    )

    return vector_store.as_retriever(search_kwargs={"k": k})
