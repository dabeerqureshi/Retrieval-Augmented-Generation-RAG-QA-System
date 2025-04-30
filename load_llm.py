from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline
import torch
import os

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Prevent TensorFlow import


# Load HuggingFace model
def load_llm_model():
    """Load a HuggingFace language model for text generation."""
    try:
        model_name = "google/flan-t5-small"
        print(f"Loading model: {model_name}")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Check for GPU availability (Colab usually has GPU)
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"
        print(f"Using device: {device_name}")

        # Create HF pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,  # Control output length
            device=device
        )

        # Wrap in LangChain
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        return None
