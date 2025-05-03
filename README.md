# Retrieval-Augmented Generation (RAG) QA System

A **RAG-based question-answering system** built using **LangChain**, **Pinecone**, and **OpenAI's Language Models**. The system processes PDF documents, splits them into text chunks, creates embeddings, and stores them in Pinecone for fast retrieval. It provides direct and relevant answers to user questions by combining document retrieval and a language model.

---

## 🚀 Features

- **PDF Document Loading**: Load PDFs containing relevant data (e.g., university prospectus).
- **Text Chunking**: Split documents into smaller chunks for efficient retrieval.
- **Embedding Creation**: Generate embeddings for text chunks using a pre-trained model.
- **Pinecone Integration**: Store embeddings in **Pinecone** for fast, scalable vector search.
- **Retrieval-Augmented QA**: Retrieve relevant document chunks and generate answers with an LLM.
- **OpenAI/Other LLM Support**: Works with various language models for text generation.
- **Prompt Template**: Custom prompts to guide the language model in generating concise and relevant answers.

---

## 🧠 How it Works

1. **Load PDF Files**: The system loads the PDFs (e.g., university prospectus).
2. **Split Text**: Text is split into smaller chunks for better processing.
3. **Create Embeddings**: Text chunks are converted into vector embeddings.
4. **Setup Pinecone**: Embeddings are stored in **Pinecone** for efficient search.
5. **Retrieve Documents**: When a query is made, the system retrieves the top relevant chunks.
6. **Generate Answer**: The language model generates answers based on retrieved context.
7. **Interactive Q&A**: Users can ask questions and receive relevant answers.

---

## ⚙️ Setup Instructions

1. Clone this repository:

   ```bash
   git clone https://github.com/dabeerqureshi/Retrieval-Augmented-Generation-RAG-QA-System.git
   cd Retrieval-Augmented-Generation-RAG-QA-System
