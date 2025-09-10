import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

BASE_MODEL_ID = "microsoft/phi-2"
ADAPTER_ID = "lakshaya17/phi2-srl"

@st.cache_resource
def load_model():
    try:
        # ✅ Load tokenizer from the adapter repo (not base!)
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)

        # ✅ Load base model (phi-2)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        # ✅ Resize base embeddings to match adapter tokenizer
        base_model.resize_token_embeddings(len(tokenizer))

        # ✅ Attach the adapter
        model = PeftModel.from_pretrained(base_model, ADAPTER_ID)

        # ✅ Wrap in HF pipeline
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    except Exception as e:
        st.error(f"🚨 Model loading failed:\n\n{e}")
        st.stop()

pipe = load_model()

st.title("🚀 Phi-2 + SRL LoRA Chatbot")

prompt = st.text_input("💬 Prompt:")
if prompt:
    with st.spinner("🧠 Thinking..."):
        try:
            result = pipe(prompt, max_new_tokens=150, temperature=0.7, do_sample=True)
            st.markdown("### ✨ Response:")
            st.write(result[0]["generated_text"][len(prompt):].strip())
        except Exception as e:
            st.error(f"⚠️ Text generation error:\n\n{e}")
