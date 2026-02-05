# retrievers.py
import mysql.connector
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_mongodb import MongoDBAtlasVectorSearch
import oracledb
from langchain_oracledb import OracleVS

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def get_mysql_retriever(host, user, password, database, port=3306):
    config = {"host": host, "user": user, "password": password, "database": database, "port": port}

    class MySQLRetriever:
        def invoke(self, query: str):
            emb = embeddings.embed_query(query)
            conn = mysql.connector.connect(**config)
            cur = conn.cursor(dictionary=True)

            sql = """
            SELECT content, metadata, (1 - (embedding <=> %s)) AS similarity
            FROM document_chunks
            ORDER BY similarity DESC
            LIMIT 6
            """
            cur.execute(sql, (str(emb),))
            rows = cur.fetchall()
            cur.close()
            conn.close()

            docs = []
            for r in rows:
                sim = float(r["similarity"])
                if sim >= 0.72:
                    docs.append(Document(
                        page_content=r["content"],
                        metadata={**(r.get("metadata") or {}), "similarity": sim, "db": "mysql"}
                    ))
            return docs

    return MySQLRetriever()


def get_postgres_retriever(connection_string: str):
    if not connection_string:
        raise ValueError("PostgreSQL connection string required")
    vs = PGVector(
        connection=connection_string,
        collection_name="rag_collection",
        embedding=embeddings,
    )
    return vs.as_retriever(search_kwargs={"k": 6})


def get_oracle_retriever(dsn, user, password):
    if not all([dsn, user, password]):
        raise ValueError("Oracle credentials incomplete")
    conn = oracledb.connect(user=user, password=password, dsn=dsn)
    vs = OracleVS(
        client=conn,
        embedding_function=embeddings,
        collection_name="RAG_DOCS",
        distance_strategy="COSINE"
    )
    return vs.as_retriever(search_kwargs={"k": 6})


def get_mongodb_retriever(uri, db_name, collection, index_name):
    if not uri:
        raise ValueError("MongoDB URI required")
    vs = MongoDBAtlasVectorSearch.from_connection_string(
        uri,
        f"{db_name}.{collection}",
        embeddings,
        index_name=index_name,
    )
    return vs.as_retriever(search_kwargs={"k": 6})