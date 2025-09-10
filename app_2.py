import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

# --- Load token from secrets ---
HF_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN")
MODEL_ID = "lakshaya17/phi2-srl"

if not HF_TOKEN:
    st.error("❌ Hugging Face token missing. Add it to your Streamlit Cloud secrets.")
    st.stop()

# Login to Hugging Face
login(HF_TOKEN)

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

llm_pipeline = load_model()

# Streamlit UI
st.title("🔬 SRCL LLM Chat Assistant")
st.markdown("Ask anything related to spacecraft proximity ops, SPOT platform, or SRCL lab content.")

user_input = st.text_area("🧠 Your question:", height=150)

if st.button("Generate Response") and user_input.strip():
    with st.spinner("Thinking..."):
        output = llm_pipeline(user_input, max_new_tokens=300, do_sample=True)[0]["generated_text"]
        st.success(output)

# Optional: Clear Chat
if st.button("🧹 Clear"):
    st.experimental_rerun()
