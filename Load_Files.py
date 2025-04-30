from langchain.text_splitter import RecursiveCharacterTextSplitter
# Updated imports for LangChain >= v0.2
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from pathlib import Path
# Extract text from PDF files
def load_pdf_files(directory_path="E:\Generative AI\Assignment#3\multimodal_rag\data"):
    """Load all PDF files from a directory."""
    path = Path(directory_path)

    if not path.exists():
        print(f"Error: Directory {directory_path} does not exist.")
        return []

    pdf_files = list(path.glob("*.pdf"))
    if not pdf_files:
        print(f"Warning: No PDF files found in {directory_path}")
        print("Please upload PDF files to the Colab runtime.")
        return []

    print(f"Found {len(pdf_files)} PDF files in {directory_path}")

    loader = DirectoryLoader(
        directory_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents

# Split documents into chunks
def split_text(documents, chunk_size=500, chunk_overlap=20):
    """Split documents into manageable chunks."""
    if not documents:
        print("No documents to split.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(documents)
    print(f"Created {len(text_chunks)} text chunks.")
    return text_chunks
