import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

# --- Page & Dark ChatGPT-Style UI Config ---
st.set_page_config(page_title="My Real Scratch AI", page_icon="🤖", layout="centered")

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
        "system_prompt": "You are a helpful, smart personal AI assistant.",
        "messages": []
    }

def save_memory(data):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

memory_data = load_memory()

# --- Load Real Open-Source AI Model from Hugging Face ---
@st.cache_resource
def load_ai_model():
    # Using Microsoft's DialoGPT - a free, lightweight conversational transformer model
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_ai_model()

# --- Sidebar: Custom Instructions & Controls ---
with st.sidebar:
    st.title("⚙️ AI Customization")
    st.caption("Give your AI personality rules or instructions.")
    
    custom_instructions = st.text_area(
        "Custom Instructions / System Prompt",
        value=memory_data["system_prompt"],
        height=150
    )
    
    if st.button("💾 Save Instructions", use_container_width=True):
        memory_data["system_prompt"] = custom_instructions
        save_memory(memory_data)
        st.success("Instructions updated!")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        memory_data["messages"] = []
        save_memory(memory_data)
        st.rerun()

# --- Main Chat Interface ---
st.title("🤖 Real Custom Generative AI")

# Display past chat history
for msg in memory_data["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input Box
if prompt := st.chat_input("Message your AI..."):
    # 1. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    memory_data["messages"].append({"role": "user", "content": prompt})
    save_memory(memory_data)

    # 2. Generate Real AI Response
    with st.chat_message("assistant"):
        with st.spinner("AI is thinking..."):
            try:
                # Encode user input and add chat history context
                new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
                
                # Append to chat history if exists
                bot_input_ids = new_user_input_ids
                
                # Generate response tokens
                chat_history_ids = model.generate(
                    bot_input_ids, 
                    max_length=1000, 
                    pad_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7
                )
                
                # Decode the response
                response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
                
                if not response.strip():
                    response = "That's interesting! Tell me more about that."
                
                st.markdown(response)
                
                # Save assistant response to memory
                memory_data["messages"].append({"role": "assistant", "content": response})
                save_memory(memory_data)
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
