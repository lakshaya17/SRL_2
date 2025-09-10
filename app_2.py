import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

# 🔧 Set your model info
BASE_MODEL_ID = "microsoft/phi-2"
ADAPTER_ID = "lakshaya17/phi2-srl"
HF_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]


@st.cache_resource
def load_model():
    try:
        # 🧠 Load tokenizer from adapter (not from base)
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID, use_auth_token=HF_TOKEN)

        # 📦 Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,         # Use float16 if GPU is available
            device_map="auto",                 # Map model across available devices
            use_auth_token=HF_TOKEN
        )

        # 🛠️ Resize embeddings to match adapter's tokenizer
        base_model.resize_token_embeddings(len(tokenizer))

        # 🔌 Load adapter and attach to base
        model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_ID,
            use_auth_token=HF_TOKEN
        )

        # ✅ Return ready-to-use generation pipeline
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    except Exception as e:
        st.error(f"🚨 Model loading failed: {e}")
        st.stop()


# Load the pipeline once
pipe = load_model()

# ──────────────────────────────────────
# 🔵 UI
st.title("🚀 Phi-2 + SRL Adapter Chatbot")
prompt = st.text_input("💬 Ask something:")

if prompt:
    with st.spinner("🧠 Generating response..."):
        try:
            output = pipe(
                prompt,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=pipe.tokenizer.eos_token_id,
            )
            generated_text = output[0]["generated_text"][len(prompt):].strip()
            st.markdown("### 📝 Response:")
            st.write(generated_text)
        except Exception as e:
            st.error(f"⚠️ Generation error: {e}")
