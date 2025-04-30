from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
import time
from pathlib import Path

PINECONE_API_KEY = "pcsk_BYY67_RBYWtDYbquoDuyg98B4t9LpPSZ28rqWtK9sT6EhGHE2yCR4rmgEN5VBrTmk3P2c"  # Replace with your key

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# Initialize or connect to Pinecone index
def setup_pinecone(index_name, dimension=384):
    """Setup Pinecone vector database."""
    try:
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("Pinecone API key not found in environment variables")

        pc = Pinecone(api_key=pinecone_api_key)

        # Check if index exists
        existing_indexes = [index['name'] for index in pc.list_indexes()]

        if index_name not in existing_indexes:
            print(f"Creating new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                )
            )
            # Wait for index to be ready
            print("Waiting for index to initialize...")
            time.sleep(10)  # Pinecone index creation takes time
        else:
            print(f"Using existing Pinecone index: {index_name}")

        return pc
    except Exception as e:
        print(f"Error setting up Pinecone: {e}")
        return None
