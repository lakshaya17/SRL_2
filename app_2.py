import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from huggingface_hub import login

# Load Hugging Face Token from secrets
HF_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN")

if not HF_TOKEN:
    st.error("❌ Hugging Face token missing. Add it to secrets.toml.")
    st.stop()

# Define model repo (replace with your actual repo if different)
MODEL_ID = "lakshaya17/phi2-srl"

# Configure Hugging Face environment (for Streamlit Cloud)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["HF_HOME"] = "/tmp/huggingface"

# Login for private repo
login(HF_TOKEN)

# Load model + tokenizer with 4-bit quantization (to fit Streamlit Cloud memory)
@st.cache_resource(show_spinner="Loading fine-tuned Phi-2 model...")
def load_model():
    quant_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        device_map="auto",
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load model
llm_pipeline = load_model()

# UI
st.title("🛰️ SRCL Fine-Tuned Phi-2 Assistant")
st.markdown("Ask questions based on the SRCL/SPOT fine-tuned model.")

user_input = st.text_input("Ask me anything:", placeholder="What is SPOT at SRCL?")

if user_input:
    with st.spinner("Generating response..."):
        response = llm_pipeline(user_input, max_new_tokens=150, do_sample=True, temperature=0.7)[0]["generated_text"]
        st.markdown("### 📤 Response")
        st.write(response.replace(user_input, "").strip())
