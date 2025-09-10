import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Model details
MODEL_ID = "lakshaya17/phi2-srl"  # Ensure this exists and is public or token-accessible
HF_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]

# Load model and tokenizer
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"🚨 Model loading failed: {e}")
        st.stop()

llm_pipeline = load_model()

# Optional chat history (enable if needed)
# if "history" not in st.session_state:
#     st.session_state.history = []

st.title("🧠 Phi-2 SRL Chat App")

user_input = st.text_input("Ask something:")

if user_input:
    with st.spinner("Generating response..."):
        try:
            result = llm_pipeline(
                user_input,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=llm_pipeline.tokenizer.eos_token_id
            )
            full_output = result[0]["generated_text"]
            generated_text = full_output[len(user_input):].strip()

            st.markdown("**Response:**")
            st.write(generated_text)

            # Optional chat history logic
            # st.session_state.history.append({"user": user_input, "bot": generated_text})
            # for msg in st.session_state.history:
            #     st.markdown(f"**You:** {msg['user']}")
            #     st.markdown(f"**Bot:** {msg['bot']}")
        except Exception as e:
            st.error(f"⚠️ Generation error: {e}")
