# RAG_SYSTEM.py

from Load_Files import load_pdf_files, split_text
from load_llm import load_llm_model
from create_embeddings import create_embeddings
from pinecone_setup import setup_pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
import os


def create_rag_system(pdf_directory="data", index_name="rag-system-index"):
    """
    Creates and returns a complete RAG system.
    """
    # 1. Load documents
    print("Loading PDF documents...")
    documents = load_pdf_files(pdf_directory)
    if not documents:
        print("No documents loaded.")
        return None

    # 2. Split into chunks
    print("Splitting documents into chunks...")
    text_chunks = split_text(documents)

    # 3. Create embeddings
    print("Creating embeddings model...")
    embeddings = create_embeddings()

    # 4. Setup Pinecone
    print("Setting up Pinecone...")
    pc = setup_pinecone(index_name)
    if not pc:
        print("Failed to setup Pinecone.")
        return None

    # 5. Store or connect to Pinecone index
    try:
        docsearch = PineconeVectorStore.from_documents(
            documents=text_chunks,
            index_name=index_name,
            embedding=embeddings,
        )
        print("Documents successfully stored in Pinecone.")
    except Exception as e:
        print(f"Error storing documents: {e}")
        print("Connecting to existing index...")
        try:
            docsearch = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=embeddings
            )
            print("Connected to existing Pinecone index.")
        except Exception as e2:
            print(f"Error connecting to Pinecone: {e2}")
            return None

    # 6. Create retriever
    retriever = docsearch.as_retriever(
        search_type='similarity',
        search_kwargs={"k": 3}
    )

    # 7. Load LLM
    print("Loading language model...")
    llm = load_llm_model()
    if not llm:
        print("Failed to load LLM.")
        return None

    # 8. Create RAG QA chain
    print("Creating RAG chain...")
    prompt_template = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to extract the most relevant answer.
    Answer the question directly and if the answer is a number or a specific fact, give it concisely.
    If you don't know the answer, say that you don't know.
    write it more carefully and when it does not know the answer say i dont know instead of wrong results

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        }
    )

    print("\n=== RAG System Ready ===")
    return qa_chain


def rag():
    return create_rag_system()
