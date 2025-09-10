import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = "lakshaya17/phi2-srl"
HF_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]

@st.cache_resource
def load_pipeline():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, use_auth_token=HF_TOKEN, torch_dtype="auto"
        )
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

pipe = load_pipeline()

st.title("🚀 Phi-2 Text Generator")

prompt = st.text_input("Enter your prompt:")
if prompt:
    with st.spinner("Generating..."):
        try:
            result = pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
            output = result[0]["generated_text"][len(prompt):].strip()
            st.markdown("**Output:**")
            st.write(output)
        except Exception as e:
            st.error(f"⚠️ Generation error: {e}")
