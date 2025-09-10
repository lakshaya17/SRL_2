import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE_MODEL_ID = "microsoft/phi-2"
ADAPTER_ID = "lakshaya17/phi2-srl"

@st.cache_resource
def load_model():
    try:
        # Load tokenizer from adapter repo
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)

        # Quantization config (4-bit)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load base model in 4-bit
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )

        # Resize embeddings to match adapter tokenizer
        base_model.resize_token_embeddings(len(tokenizer))

        # Load adapter
        model = PeftModel.from_pretrained(base_model, ADAPTER_ID)

        # Build pipeline
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    except Exception as e:
        st.error(f"🚨 Model loading failed:\n\n{e}")
        st.stop()

# Load pipeline once
pipe = load_model()

# ──────────────── UI ────────────────
st.title("🚀 Phi-2 + SRL Adapter (4-bit LoRA)")

prompt = st.text_input("💬 Enter your prompt:")
if prompt:
    with st.spinner("🧠 Thinking..."):
        try:
            result = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
            response = result[0]["generated_text"][len(prompt):].strip()
            st.markdown("### ✨ Response:")
            st.write(response)
        except Exception as e:
            st.error(f"⚠️ Text generation failed:\n\n{e}")
