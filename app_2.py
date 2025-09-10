import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from dotenv import load_dotenv

# Load token from Streamlit secrets or .env
HF_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", None)
if HF_TOKEN is None:
    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if HF_TOKEN is None:
    st.error("❌ Hugging Face token not found. Add HUGGINGFACE_TOKEN to Streamlit secrets.")
    st.stop()

# Login to HF Hub
login(HF_TOKEN)

# Load model and tokenizer
MODEL_ID = "lakshaya17/phi2-srl"

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

llm_pipeline = load_model()

# Streamlit UI
st.set_page_config(page_title="SRCL LLM Chat", page_icon="🛰️", layout="wide")
st.title("🛰️ Fine-tuned Phi-2 LLM Chat (SRCL)")

# Session state to maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# Chat input
user_input = st.chat_input("Ask something about SRCL, SPOT, or spacecraft systems...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # You can adjust max_new_tokens and temperature
            response = llm_pipeline(user_input, max_new_tokens=300, do_sample=True, temperature=0.7)[0]["generated_text"]
            cleaned_response = response[len(user_input):].strip()
            st.markdown(cleaned_response)
            st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
