# ✅ Must be the first Streamlit command
import streamlit as st
st.set_page_config(page_title="RAG QA System", layout="centered")

# Now import everything else
from Query import query_rag
from metrics import bleu_score, rouge_score, cosine_sim, response_time
import easyocr
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Optional: Dummy embeddings for cosine similarity demo
# In practice, use actual vector outputs from your embedding models
dummy_gt_embedding = [0.1, 0.2, 0.3]
dummy_ans_embedding = [0.1, 0.22, 0.28]

# Initialize EasyOCR reader (for OCR in case image has text)
ocr_reader = easyocr.Reader(['en'])

# Initialize image model (ResNet example for image embeddings)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model = model.to(device)
model.eval()

# Streamlit UI
st.title("📚 Retrieval-Augmented Generation (RAG) QA System")

# Input section
question = st.text_input("🔍 Ask your question:")
reference_answer = st.text_input("✅ Ground-truth answer (optional for evaluation):")

# Image input section
image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Image embedding extraction function
def extract_image_embedding(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.cpu().numpy().flatten()

# Text or image query handling
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

# Image-based query handling
if image_file:
    with st.spinner("Processing image..."):
        image_embedding = extract_image_embedding(image_file)

        # If you have an image-based RAG model, you can query it similarly to the text-based query
        # For example, let's assume you query a system with an image embedding (you can replace this with actual query logic)
        result, latency = response_time(query_rag, image_embedding)

    # Display answer for image query
    st.subheader("💡 Answer from Image:")
    st.markdown(result["answer"])

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
