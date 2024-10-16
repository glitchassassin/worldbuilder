from typing import List
from pymilvus import MilvusClient, model

# Initialize Milvus connection
client = MilvusClient("milvus.db")

embedding_fn = model.DefaultEmbeddingFunction()

# Create collection
if not client.has_collection("documents"):
    client.create_collection(collection_name="documents", dimension=768)

def store_document(content: str):
    """
    Stores a document and its embedding in Milvus.
    """
    embedding = embedding_fn.encode_documents([content])
    data = [
        {
            "id": 1,
            "vector": embedding[0],
            "text": content
        }
    ]
    client.insert(collection_name="documents", data=data)

def query_documents(query: str, top_k: int = 10) -> List[str]:
    """
    Queries Milvus for the top_k most similar documents based on the embedding.
    """
    results = client.search(
        collection_name="documents",
        data=embedding_fn.encode_queries([query]),
        limit=top_k,
        output_fields=["text"]
    )
    return [hit["entity"].get("text") for hit in results[0]]