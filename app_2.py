import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = "lakshaya17/phi2-srl"
HF_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

llm_pipeline = load_model()

user_input = st.text_input("Ask something:")
if user_input:
    with st.spinner("Generating..."):
        result = llm_pipeline(user_input, max_new_tokens=200, do_sample=True, temperature=0.7)
        st.write(result[0]['generated_text'])
