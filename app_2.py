import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

BASE_MODEL_ID = "microsoft/phi-2"
ADAPTER_ID = "lakshaya17/phi2-srl"

@st.cache_resource
def load_model():
    try:
        # Load tokenizer from adapter repo
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)

        # Load base model (phi-2)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,  # use float32 if you're on CPU
            device_map="auto"
        )

        # Resize base model embeddings to match tokenizer
        base_model.resize_token_embeddings(len(tokenizer))

        # Load and attach adapter
        model = PeftModel.from_pretrained(base_model, ADAPTER_ID)

        # Create text-generation pipeline
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    except Exception as e:
        st.error(f"🚨 Model loading failed: {e}")
        st.stop()

# Load model once
pipe = load_model()

# ────────────────────────────────
st.title("🧠 Phi-2 + SRL Adapter (LoRA)")

prompt = st.text_input("💬 Enter your prompt:")

if prompt:
    with st.spinner("Generating response..."):
        try:
            result = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
            generated_text = result[0]["generated_text"][len(prompt):].strip()
            st.markdown("### ✨ Response:")
            st.write(generated_text)
        except Exception as e:
            st.error(f"⚠️ Generation failed: {e}")
