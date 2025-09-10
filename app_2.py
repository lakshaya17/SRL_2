import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set page config
st.set_page_config(page_title="SRCL Q&A", layout="centered")

# Title
st.title("🤖 Ask the SRCL Trained LLM")
st.markdown("Ask any question based on the fine-tuned SPOT/SRCL model.")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = "models/phi2_srcl_lora"  # Replace with your actual path or HuggingFace repo
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    return tokenizer, model

tokenizer, model = load_model()

# Input box
user_input = st.text_input("Your question:", "")

# Generate response
if st.button("Generate Answer"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        output = model.generate(input_ids, max_new_tokens=256, do_sample=True)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Clean output
        cleaned = response.replace(user_input, "").strip()
        st.markdown("**Answer:**")
        st.write(cleaned)

