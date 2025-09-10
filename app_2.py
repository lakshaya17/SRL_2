import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load fine-tuned Phi-2 model from Hugging Face
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("lakshaya17/phi2-srl")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "your-username/your-finetuned-phi2",
        torch_dtype=torch.float32,  # use float16 if GPU supports
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.set_page_config(page_title="SRCL Fine-Tuned Chatbot", layout="wide")
st.title("🚀 SRCL Fine-Tuned Phi-2 Chatbot")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Clear chat button
with st.sidebar:
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask something about SRCL / SPOT lab...")
if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            inputs = tokenizer.encode(user_input, return_tensors="pt").to(model.device)
            output = model.generate(
                inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_k=40,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            # Trim prompt from response
            response = response[len(user_input):].strip()
            st.markdown(response)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
