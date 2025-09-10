import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

BASE_MODEL_ID = "microsoft/phi-2"
ADAPTER_ID = "lakshaya17/phi2-srl"
HF_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]

@st.cache_resource
def load_model():
    try:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=HF_TOKEN
        )

        # Load adapter
        model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_ID,
            use_auth_token=HF_TOKEN
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_auth_token=HF_TOKEN)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    except Exception as e:
        st.error(f"🚨 Model loading failed: {e}")
        st.stop()

pipe = load_model()

st.title("🚀 Phi-2 + SRL Adapter Chat")

prompt = st.text_input("Enter your prompt:")
if prompt:
    with st.spinner("Generating..."):
        try:
            output = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
            generated_text = output[0]["generated_text"][len(prompt):].strip()
            st.markdown("**Response:**")
            st.write(generated_text)
        except Exception as e:
            st.error(f"⚠️ Generation error: {e}")
