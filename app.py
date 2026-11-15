import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

# --- Page & Dark ChatGPT-Style UI Config ---
st.set_page_config(page_title="My Custom AI", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #212121; color: #ECECEC; }
    [data-testid="stSidebar"] { background-color: #171717; border-right: 1px solid #2F2F2F; }
    .stChatMessage { background-color: #2F2F2F; border-radius: 14px; padding: 12px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

MEMORY_FILE = "memory.json"

# --- Load Memory & Custom Instructions ---
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "system_prompt": "You are a smart, direct, and helpful personal AI assistant.",
        "messages": []
    }

def save_memory(data):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

memory_data = load_memory()

# --- Load Real Instruction-Tuned AI Model ---
@st.cache_resource
def load_ai_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )
    return tokenizer, model

tokenizer, model = load_ai_model()

# --- Sidebar: Custom Instructions & Controls ---
with st.sidebar:
    st.title("⚙️ AI Customization")
    
    custom_instructions = st.text_area(
        "Custom Instructions / Personality",
        value=memory_data["system_prompt"],
        height=150
    )
    
    if st.button("💾 Save Instructions", use_container_width=True):
        memory_data["system_prompt"] = custom_instructions
        save_memory(memory_data)
        st.success("Instructions saved!")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        memory_data["messages"] = []
        save_memory(memory_data)
        st.rerun()

# --- Main Chat Interface ---
st.title("🧠 Custom AI Assistant")

for msg in memory_data["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Message your AI..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    memory_data["messages"].append({"role": "user", "content": prompt})
    save_memory(memory_data)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Build conversation with your custom instructions + recent chat memory
                conversation = [{"role": "system", "content": memory_data["system_prompt"]}]
                conversation.extend(memory_data["messages"][-10:])  # Keeps last 10 messages in context

                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt")

                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True
                )
                
                # Trim out the prompt so only the AI's new reply shows
                generated_ids = [
                    output_ids[len(input_ids):] 
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                st.markdown(response)
                memory_data["messages"].append({"role": "assistant", "content": response})
                save_memory(memory_data)

            except Exception as e:
                st.error(f"Error: {e}")
