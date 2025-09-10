import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

# Public model IDs
BASE_MODEL_ID = "microsoft/phi-2"
ADAPTER_MODEL_ID = "lakshaya17/phi2-srl"

@st.cache_resource
def load_pipeline():
    try:
        # Load tokenizer from adapter repo
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL_ID)

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        # Resize embeddings to match adapter tokenizer
        base_model.resize_token_embeddings(len(tokenizer))

        # Load adapter (LoRA)
        model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_ID)

        # Return text-generation pipeline
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    except Exception as e:
        st.error(f"🚨 Model loading failed:\n\n{e}")
        st.stop()

# Load the model once
pipe = load_pipeline()

# ──────────────── Streamlit UI ────────────────
st.title("🚀 Phi-2 + SRL Adapter (LoRA Chat)")

prompt = st.text_input("💬 Enter your prompt:")
if prompt:
    with st.spinner("Generating response..."):
        try:
            result = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
            response = result[0]["generated_text"][len(prompt):].strip()
            st.markdown("### ✨ Response:")
            st.write(response)
        except Exception as e:
            st.error(f"⚠️ Text generation failed:\n\n{e}")
