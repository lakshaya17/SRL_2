# Requirements (put these in requirements.txt if you haven't already):
# streamlit
# requests
# beautifulsoup4
# trafilatura
# python-dotenv
# google-generativeai          # only if you want Gemini option
# transformers
# peft
# bitsandbytes                 # for 8-bit loading (optional but recommended)
#
# Run: streamlit run app_gemini.py

import os
import time
import streamlit as st
import requests
from bs4 import BeautifulSoup
import trafilatura
from dotenv import load_dotenv

# Optional Gemini imports (we handle missing package gracefully)
try:
    import google.generativeai as genai
    HAVE_GEMINI = True
except Exception:
    HAVE_GEMINI = False

# Local LLM imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# -------------------------------
# Config & Hardcoded URLs
# -------------------------------
URL_LIST = [
    "https://github.com/Carleton-SRL/SPOT/wiki/Overview-of-the-Spacecraft-Proximity-Operations-Testbed",
    "https://github.com/Carleton-SRL/SPOT/wiki/Notation-and-Standard-Nomenclature",
    "https://github.com/Carleton-SRL/SPOT/wiki/Reference-Frame-Definitions",
    "https://github.com/Carleton-SRL/SPOT/wiki/Linux-Development-Station",
    "https://github.com/Carleton-SRL/SPOT/wiki/Using-the-3D-Printer-–-Rules-and-Recommendations",
    "https://github.com/Carleton-SRL/SPOT/wiki/First-Time-Software-Setup",
    "https://github.com/Carleton-SRL/SPOT/wiki/Running-a-Simulation",
    "https://github.com/Carleton-SRL/SPOT/wiki/Experiment-Booking-Logistics",
    "https://github.com/Carleton-SRL/SPOT/wiki/Setting-up-the-Laboratory-for-an-Experiment",
    "https://github.com/Carleton-SRL/SPOT/wiki/Running-an-Experiment",
    "https://github.com/Carleton-SRL/SPOT/wiki/Platform-Shutdown-Procedure",
    "https://github.com/Carleton-SRL/SPOT/wiki/End-of-Day-Procedures",
    "https://github.com/Carleton-SRL/SPOT/wiki/Emergency-Procedures",
    "https://github.com/Carleton-SRL/SPOT/wiki/Compressor-Instructions",
    "https://github.com/Carleton-SRL/SPOT/wiki/Spacecraft-Assembly-Disassembly-Procedure",
    "https://github.com/Carleton-SRL/SPOT/wiki/Measuring-the-Mass-Properties",
    "https://github.com/Carleton-SRL/SPOT/wiki/Exporting-Experiment-Videos-with-QC-Viewer",
    "https://github.com/Carleton-SRL/SPOT/wiki/PhaseSpace-Calibration-Procedure",
    "https://github.com/Carleton-SRL/SPOT/wiki/Re-assigning-or-Setting-up-a-PhaseSpace-LED-Driver",
    "https://github.com/Carleton-SRL/SPOT/wiki/How-to-Submit-a-GitHub-Issue",
    "https://github.com/Carleton-SRL/SPOT/wiki/Remoting-into-the-a-Xavier-Computer",
    "https://github.com/Carleton-SRL/SPOT/wiki/Manually-Controlling-GPIO-Pins-(Pucks)",
    "https://github.com/Carleton-SRL/SPOT/wiki/Copying-Files-between-Windows-and-Linux",
    "https://github.com/Carleton-SRL/SPOT/wiki/Lab-IP-Addresses-&-Router-Login",
    "https://github.com/Carleton-SRL/SPOT/wiki/Assigning-an-IP-Address-to-a-New-Device",
    "https://github.com/Carleton-SRL/SPOT/wiki/Setting-up-SSH-Key-for-Passwordless-File-Transfers",
    "https://github.com/Carleton-SRL/SPOT/wiki/Getting-Started-with-Computer-Vision",
    "https://github.com/Carleton-SRL/SPOT/wiki/Getting-Started-with-the-Jetson-Computer",
    "https://github.com/Carleton-SRL/SPOT/wiki/Sending-Data-to-a-Jetson-Computer",
    "https://github.com/Carleton-SRL/SPOT/wiki/Sending-Data-from-the-Jetson-Orin-to-Simulink",
    "https://github.com/Carleton-SRL/SPOT/wiki/First-Time-ZED2-Camera-Setup",
    "https://github.com/Carleton-SRL/SPOT/wiki/Recording-ZED2-Camera-Footage",
    "https://github.com/Carleton-SRL/SPOT/wiki/Setup-&-Use-of-Pose-Detection-CNN-(ZED2)",
    "https://github.com/Carleton-SRL/SPOT/wiki/Getting-Started-with-the-LiDAR-Camera-L515",
    "https://github.com/Carleton-SRL/SPOT/wiki/Setting-up-the-Software-Requirements-to-use-LiDAR-for-Pose-Determination",
    "https://github.com/Carleton-SRL/SPOT/wiki/Getting-Started-with-the-Garmin-Lidar-Lite-v3",
    "https://github.com/Carleton-SRL/SPOT/wiki/Flashing-a-Jetson-Computer",
    "https://github.com/Carleton-SRL/SPOT/wiki/Installing-Drivers-for-the-TP-LINK-TL-WN722N",
    "https://github.com/Carleton-SRL/SPOT/wiki/PCA9685-Code-Overview-(Hardware-PWM)",
    "https://github.com/Carleton-SRL/SPOT/wiki/Using-Jetson.GPIO-for-PWM-(Software-PWM)",
    "https://github.com/Carleton-SRL/SPOT/wiki/Simulink-Diagram-Details",
    "https://github.com/Carleton-SRL/SPOT/wiki/Graphical-User-Interface-Details-(WIP)",
    "https://github.com/Carleton-SRL/SPOT/wiki/Running-Simulations-without-the-GUI",
    "https://github.com/Carleton-SRL/SPOT/wiki/Dynamixel-Actuators-Code-Overview",
    "https://github.com/Carleton-SRL/SPOT/wiki/Download-the-PhaseSpace-X2E-SDK-API",
    "https://github.com/Carleton-SRL/SPOT/wiki/Creating-a-Custom-Device-Driver-Block-(Source)",
    "https://github.com/Carleton-SRL/SPOT/wiki/Creating-a-Custom-Device-Driver-Block-(Sink)",
    "https://github.com/Carleton-SRL/SPOT/wiki/Collecting-Raw-PhaseSpace-Data-(C++)",
    "https://github.com/Carleton-SRL/SPOT/wiki/Structure-Details",
    "https://github.com/Carleton-SRL/SPOT/wiki/Docking-Specifications",
    "https://github.com/Carleton-SRL/SPOT/wiki/Propulsion-Subsystem",
    "https://github.com/Carleton-SRL/SPOT/wiki/Flotation-Subsystem",
    "https://github.com/Carleton-SRL/SPOT/wiki/Power-Subsystem",
    "https://github.com/Carleton-SRL/SPOT/wiki/Vision-Subsystem",
    "https://github.com/Carleton-SRL/SPOT/wiki/Reaction-Wheel-Subsystem",
    "https://github.com/Carleton-SRL/SPOT/wiki/Robotic-Manipulator-Subsystem",
    "https://github.com/Carleton-SRL/SPOT/wiki/Emergency-Stop-System",
    "https://github.com/Carleton-SRL/SPOT/wiki/Ground-Truth-System",
    "https://github.com/Carleton-SRL/SPOT/wiki/Computer-Subsystem",
    "https://github.com/Carleton-SRL/SPOT/wiki/Hardware-PWM-Subsystem",
    "https://github.com/Carleton-SRL/SPOT/wiki/Solar-Panel-Subsystem",
    "https://github.com/Carleton-SRL/SPOT/wiki/Reimbursement-Process-for-Out-of-Pocket-Expenses",
    "https://github.com/Carleton-SRL/SPOT/wiki/SPOT-Velocity-Estimator-–-Summer-2024-Showdown",
    "https://github.com/Carleton-SRL/SPOT/wiki/Regarding-MacOS-Compatibility",
    "https://github.com/Carleton-SRL/SPOT/wiki/About-Branches-&-Forks",
    "https://github.com/Carleton-SRL/SPOT/wiki/Creating-a-Fork",
    "https://github.com/Carleton-SRL/SPOT/wiki/Updating-a-Fork",
    "https://github.com/Carleton-SRL/SPOT/wiki/Creating-a-Pull-Request-(Fork)",
    "https://github.com/Carleton-SRL/SPOT/wiki/Creating-a-Branch",
    "https://github.com/Carleton-SRL/SPOT/wiki/Updating-a-Branch",
    "https://github.com/Carleton-SRL/SPOT/wiki/Creating-a-Pull-Request-(Branch)",
    "https://github.com/Carleton-SRL/SPOT/wiki/Software-Issue-–-Debugging-the-SPOT-GUI",
    "https://github.com/Carleton-SRL/SPOT/wiki/Software-Issue-–-Custom-Library-Error",
    "https://github.com/Carleton-SRL/SPOT/wiki/Experiment-Issue-–-PhaseSpace-Turns-on-Briefly,-then-Turns-Off",
    "https://github.com/Carleton-SRL/SPOT/wiki/Experiment-Issue-–-PhaseSpace-does-not-turn-on-when-clicking-Start-Experiment",
    "https://github.com/Carleton-SRL/SPOT/wiki/Experiment-Issue-–-Experiment-not-Starting-but-LEDs-are-ON",
    "https://github.com/Carleton-SRL/SPOT/wiki/Experiment-Issue-–-Unstable-Platforms",
    "https://github.com/Carleton-SRL/SPOT/wiki/Experiment-Issue-–-LEDs-on-platform(s)-not-turning-on-or-finicky-connection",
    "https://github.com/Carleton-SRL/SPOT/wiki/Experiment-Issue-–-It-is-difficult-to-remove-the-air-tank-from-the-platform",
    "https://github.com/Carleton-SRL/SPOT/wiki/Hardware-Issue-–-Pucks-not-turning-on",
    "https://github.com/Carleton-SRL/SPOT/wiki/Hardware-Issue-–-MX-64-Actuators-not-sending-Encoder-Data",
    "https://github.com/Carleton-SRL/SPOT/wiki/Hardware-Issue-–-One-or-More-Thrusters-are-Stuck-Open",
    "https://github.com/Carleton-SRL/SPOT/wiki/Hardware-Issue-–-The-Intel-RealSense-L515-is-Not-Working-on-the-Jetson-Orin",
    "https://github.com/Carleton-SRL/SPOT/wiki/Google-Photos",
    "https://github.com/Carleton-SRL/SPOT/wiki/YouTube-Channel",
    "https://carleton.ca/spacecraft/news/",
    "https://carleton.ca/spacecraft/",
    "https://carleton.ca/spacecraft/research/",
    "https://carleton.ca/spacecraft/datasets/",
    "https://carleton.ca/spacecraft/media/",
    "https://carleton.ca/spacecraft/teaching/",
    "https://carleton.ca/spacecraft/contact/",
    "https://carleton.ca/spacecraft/members/",
    "https://carleton.ca/spacecraft/publications/",
    "https://carleton.ca/spacecraft/about/",
    "https://carleton.ca/spacecraft/partners/",
    "https://carleton.ca/spacecraft/opportunities/"
]

DEFAULT_MAX_CONTEXT_CHARS = 24000
LOCAL_MODEL_NAME = "microsoft/phi-2"        # base
LOCAL_ADAPTER_DIR = "models/phi2_srcl_lora"  # your LoRA adapter (unzipped folder)

# --- PAGE CONFIGURATION
st.set_page_config(page_title="Spacecraft Robotics Laboratory AI Assistant", layout="wide", initial_sidebar_state="collapsed")
st.title("🛰️ Spacecraft Robotics Laboratory AI Assistant")

# -------------------------------
# Session state
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Clear button
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# -------------------------------
# Sidebar: engine & settings
# -------------------------------
with st.sidebar:
    st.markdown("### Engine")
    engine = st.radio("Choose engine", options=["Local SRCL (Phi-2 LoRA)", "Gemini 1.5 Flash"], index=0)
    max_chars = st.slider("Max context characters", 4000, 80000, DEFAULT_MAX_CONTEXT_CHARS, 1000)
    st.caption("Smaller = faster; Larger = more grounding text.")

# -------------------------------
# Utilities
# -------------------------------
def fetch_clean(url: str, timeout: int = 20):
    """Fetch URL and extract clean text with trafilatura fallback to BS4."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        extracted = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        if extracted and len(extracted.strip()) > 200:
            return extracted.strip()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.body or soup
        text = main.get_text(separator="\n")
        text = "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])
        return text
    except Exception as e:
        return f"[ERROR fetching {url}]: {e}"

def build_context(pages, cap_chars: int):
    """Concatenate retrieved page texts with separators; cap size."""
    ctx_chunks, total = [], 0
    for p in pages:
        header = f"\n\n==== SOURCE: {p['url']} ====\n"
        chunk = header + (p.get("text") or "")
        if total + len(chunk) > cap_chars:
            remaining = cap_chars - total
            if remaining > 0:
                ctx_chunks.append(chunk[:remaining])
            break
        ctx_chunks.append(chunk)
        total += len(chunk)
    return "".join(ctx_chunks)

# -------------------------------
# Local LLM (Phi-2 + LoRA) cache
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_local_pipeline():
    # Load tokenizer + base model; try 8-bit first, fallback to fp16/cpu if needed
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_NAME,
            device_map="auto",
            load_in_8bit=True
        )
    except Exception:
        # Fallback if bitsandbytes not available
        base_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_NAME,
            device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
    # Attach LoRA adapter
    if not os.path.isdir(LOCAL_ADAPTER_DIR):
        raise FileNotFoundError(
            f"LoRA adapter not found at '{LOCAL_ADAPTER_DIR}'. "
            "Unzip your fine-tuned adapter folder there."
        )
    ft_model = PeftModel.from_pretrained(base_model, LOCAL_ADAPTER_DIR)
    # Build HF pipeline
    pipe = pipeline(
        "text-generation",
        model=ft_model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1
    )
    return pipe

def answer_with_local(question: str, context: str):
    pipe = load_local_pipeline()
    prompt = (
        "You are an assistant for Carleton University's SRCL. "
        "Use ONLY the provided context to answer. If not in context, say you don't know.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
    )
    out = pipe(prompt)[0]["generated_text"]
    # Return the part after "ANSWER:" to be neat
    return out.split("ANSWER:", 1)[-1].strip()

# -------------------------------
# Gemini (optional) setup
# -------------------------------
def maybe_init_gemini():
    if not HAVE_GEMINI:
        st.error("google-generativeai is not installed. Install it or switch to Local SRCL engine.")
        st.stop()
    if not GEMINI_KEY:
        st.error("GEMINI_API_KEY not found in .env. Set it or switch to Local SRCL engine.")
        st.stop()
    genai.configure(api_key=GEMINI_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")

def answer_with_gemini(model, question: str, context: str):
    system_instructions = (
        "You are an assistant for Carleton University's SRCL. "
        "Use ONLY the provided context to answer. If unsure, say you don't know. "
        "Keep answers concise and cite sources as [n] with URLs listed afterward."
    )
    prompt = (
        f"{system_instructions}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "RESPONSE FORMAT:\n"
        "1-2 paragraphs. Add inline citations like [1], [2] where relevant. "
        "Then list Sources as a numbered list with URLs."
    )
    resp = model.generate_content(prompt)
    return resp.text if hasattr(resp, "text") else str(resp)

# -------------------------------
# Main UI
# -------------------------------
query = st.text_input("Ask a question about SRCL (research, SPOT, procedures, hardware, etc.):")

if query:
    with st.spinner("📚 Collecting pages..."):
        pages = []
        for url in URL_LIST:
            text = fetch_clean(url)
            pages.append({"url": url, "text": text})
            time.sleep(0.2)  # basic politeness

    context = build_context(pages, cap_chars=max_chars)

    with st.spinner("🤖 Generating answer..."):
        if engine == "Local SRCL (Phi-2 LoRA)":
            try:
                answer = answer_with_local(query, context)
            except Exception as e:
                st.error(f"Local model error: {e}")
                answer = ""
        else:
            gem = maybe_init_gemini()
            answer = answer_with_gemini(gem, query, context)

    if answer:
        st.session_state.chat_history.append((query, answer))

# -------------------------------
# Chat History Display
# -------------------------------
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("## 📝 Chat History")
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")
        st.markdown("---")
