import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Hugging Face token (from Streamlit Cloud secrets or .env)
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# App title
st.set_page_config(page_title="Chat with Fine-Tuned Phi-2")
st.title("🚀 SRCL Fine-Tuned Phi-2 Chatbot")

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "lakshaya17/phi2-srl",
        token=HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "lakshaya17/phi2-srl",
        token=HF_TOKEN,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

# Session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about SPOT, SRCL, or your setup..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()
