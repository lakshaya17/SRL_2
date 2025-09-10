import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

BASE_MODEL_ID = "microsoft/phi-2"
ADAPTER_ID = "lakshaya17/phi2-srl"

@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"🚨 Model loading failed: {e}")
        st.stop()

pipe = load_model()

st.title("🧠 Phi-2 + SRL Adapter (LoRA)")

prompt = st.text_input("💬 Enter your prompt:")
if prompt:
    with st.spinner("Generating..."):
        try:
            result = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
            output = result[0]["generated_text"][len(prompt):].strip()
            st.markdown("### ✨ Response:")
            st.write(output)
        except Exception as e:
            st.error(f"⚠️ Generation failed: {e}")
