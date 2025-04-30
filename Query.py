# Query.py

import streamlit as st
from RAG_SYSTEM import rag

# Cache the RAG system initialization to avoid reloading on every question
@st.cache_resource(show_spinner="Initializing RAG system...")
def get_cached_rag_system():
    return rag()

rag_system = get_cached_rag_system()


# Function for querying the RAG system
def query_rag(question, rag_chain=rag_system):
    """Query the RAG system with a question."""
    if not question.strip():
        return {"answer": "Please enter a valid question.", "sources": []}

    if rag_chain is None:
        return {"answer": "RAG system not initialized. Please run the setup first.", "sources": []}

    try:
        result = rag_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.page_content[:150] + "..." for doc in result["source_documents"]]
        }
    except Exception as e:
        return {"answer": f"Error processing query: {str(e)}", "sources": []}
