from langchain.embeddings import HuggingFaceEmbeddings
# Create embeddings model
def create_embeddings():
    """Create a HuggingFace embeddings model."""
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    print("Embeddings model loaded successfully")
    return embeddings
