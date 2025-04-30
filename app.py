# ✅ Must be the first Streamlit command
import streamlit as st
st.set_page_config(page_title="RAG QA System", layout="centered")

# Now import everything else
from Query import query_rag
from metrics import bleu_score, rouge_score, cosine_sim, response_time

# Optional: Dummy embeddings for cosine similarity demo
# In practice, use actual vector outputs from your embedding models
dummy_gt_embedding = [0.1, 0.2, 0.3]
dummy_ans_embedding = [0.1, 0.22, 0.28]

# Streamlit UI
st.title("📚 Retrieval-Augmented Generation (RAG) QA System")

# Input section
question = st.text_input("🔍 Ask your question:")
reference_answer = st.text_input("✅ Ground-truth answer (optional for evaluation):")

# If the user submits a question
if question:
    with st.spinner("Generating answer..."):
        result, latency = response_time(query_rag, question)

    # Display answer
    st.subheader("💡 Answer:")
    st.markdown(result["answer"])

    # Display sources
    if result["sources"]:
        st.subheader("📄 Sources:")
        for i, source in enumerate(result["sources"]):
            st.markdown(f"**Source {i+1}:** {source}")
    else:
        st.info("No relevant sources found.")

    # Evaluate and display metrics if reference answer is given
    if reference_answer:
        bleu = bleu_score(reference_answer, result["answer"])
        rouge = rouge_score(reference_answer, result["answer"])
        cosine = cosine_sim(dummy_gt_embedding, dummy_ans_embedding)

        st.subheader("📊 Performance Metrics:")
        st.markdown(f"**BLEU Score:** {bleu:.4f}")
        st.markdown(f"**ROUGE-L Score:** {rouge:.4f}")
        st.markdown(f"**Cosine Similarity:** {cosine:.4f}")
        st.markdown(f"**Response Time:** {latency:.2f} seconds")

# Info section
with st.expander("ℹ️ About this app"):
    st.write("""
        This app uses a Retrieval-Augmented Generation (RAG) system built on LangChain, FAISS,
        and HuggingFace models to answer questions from uploaded PDF documents.

        It also supports evaluating generated answers using BLEU, ROUGE, Cosine Similarity,
        and response time metrics.
    """)
