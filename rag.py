from db.milvus import query_documents
import ell

@ell.simple(model="gpt-4o-mini")
def rag_query(query: str) -> str:
    """You are a helpful assistant."""
    relevant_docs = query_documents(query, top_k=5)
    context = "\n".join(relevant_docs)
    
    return f"Query: {query}\n\nContext:\n{context}"