from db.milvus import store_document, query_documents
import openai
import os
from dotenv import load_dotenv
from rag import rag_query
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def add_document(content: str):
    store_document(content)

def get_similar_documents(query: str):
    return query_documents(query)

add_document("The secret word is: applesauce")
print(rag_query("What is the secret word?"))