import streamlit as st
from RAG_SYSTEM import rag
import numpy as np  # To handle image embeddings

# Cache the RAG system initialization to avoid reloading on every question
@st.cache_resource(show_spinner="Initializing RAG system...")
def get_cached_rag_system():
    return rag()

rag_system = get_cached_rag_system()


# Function for querying the RAG system
def query_rag(query, rag_chain=rag_system):
    """Query the RAG system with a text query or image embedding."""
    if isinstance(query, str):  # Handle text-based queries
        if not query.strip():
            return {"answer": "Please enter a valid question.", "sources": []}

        if rag_chain is None:
            return {"answer": "RAG system not initialized. Please run the setup first.", "sources": []}

        try:
            result = rag_chain({"query": query})
            return {
                "answer": result["result"],
                "sources": [doc.page_content[:150] + "..." for doc in result["source_documents"]]
            }
        except Exception as e:
            return {"answer": f"Error processing query: {str(e)}", "sources": []}

    elif isinstance(query, np.ndarray):  # Handle image-based queries (embedding)
        if rag_chain is None:
            return {"answer": "RAG system not initialized. Please run the setup first.", "sources": []}

        # If the RAG system expects a string, we need to convert the embedding to a string or handle appropriately.
        try:
            # Convert the image embedding to a suitable string (or modify this based on your RAG system)
            embedding_str = " ".join(map(str, query.flatten()))  # Convert the image embedding to a space-separated string
            
            # Query the RAG system with the image embedding string
            result = rag_chain({"query": embedding_str})
            
            return {
                "answer": result["result"],
                "sources": [doc.page_content[:150] + "..." for doc in result["source_documents"]]
            }
        except Exception as e:
            return {"answer": f"Error processing image query: {str(e)}", "sources": []}

    else:
        return {"answer": "Unsupported query type", "sources": []}
