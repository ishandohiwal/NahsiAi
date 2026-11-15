import streamlit as st
import torch
import json
import random
import os
import time
from model import CustomBrainNet
from train import bag_of_words, tokenize

# --- Page & Dark ChatGPT-Style UI Config ---
st.set_page_config(page_title="My Scratch AI", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #212121; color: #ECECEC; }
    [data-testid="stSidebar"] { background-color: #171717; border-right: 1px solid #2F2F2F; }
    .stChatMessage { background-color: #2F2F2F; border-radius: 14px; padding: 12px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

MEMORY_FILE = "memory.json"

# --- Load Memory & Personalization ---
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"user_name": "Boss", "ai_vibe": "direct and helpful", "custom_facts": [], "messages": []}

def save_memory(data):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

memory = load_memory()

# --- Load Your Trained PyTorch Neural Network ---
@st.cache_resource
def load_brain():
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
    data = torch.load("brain.pth", map_location=torch.device('cpu'))
    model = CustomBrainNet(data["input_size"], data["hidden_size"], data["output_size"])
    model.load_state_dict(data["model_state"])
    model.eval()
    return model, data["all_words"], data["tags"], intents

if not os.path.exists("brain.pth"):
    st.error("⚠️ `brain.pth` not found! Run `python train.py` in your terminal first to train your AI.")
    st.stop()

model, all_words, tags, intents = load_brain()

# --- Sidebar: Personalization & Memory Bank ---
with st.sidebar:
    st.title("🧠 AI Customization")
    
    new_name = st.text_input("What should the AI call you?", value=memory["user_name"])
    new_vibe = st.text_area("Personality / Custom Instructions", value=memory["ai_vibe"], height=100)
    
    if st.button("💾 Save Customization", use_container_width=True):
        memory["user_name"] = new_name
        memory["ai_vibe"] = new_vibe
        save_memory(memory)
        st.success("Personalization saved!")

    st.divider()
    st.subheader("📌 Permanent Memory Bank")
    st.caption("Tip: Type 'remember that [fact]' in chat to teach the AI new facts.")
    if memory["custom_facts"]:
        for fact in memory["custom_facts"]:
            st.markdown(f"- {fact}")
    else:
        st.caption("No facts saved yet.")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        memory["messages"] = []
        save_memory(memory)
        st.rerun()

# --- Main Chat Screen ---
st.title("🧠 Custom Scratch AI")

for msg in memory["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Neural Inference + Memory Logic ---
def generate_reply(user_text):
    lower_text = user_text.lower()

    # 1. Active Memory Learning ("remember that...")
    if "remember that " in lower_text:
        fact = user_text.split("remember that ", 1)[1].strip()
        memory["custom_facts"].append(fact)
        save_memory(memory)
        return f"Got it, {memory['user_name']}. Saved to my permanent memory: *\"{fact}\"*"

    # 2. Run Forward Pass Through Your PyTorch Neural Network
    tokens = tokenize(user_text)
    X = bag_of_words(tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    confidence = probs[0][predicted.item()].item()

    # 3. Pick Response & Apply Personalization
    facts_str = ", ".join(memory["custom_facts"]) if memory["custom_facts"] else "nothing saved yet"
    
    if confidence > 0.70:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                raw_reply = random.choice(intent["responses"])
                return raw_reply.format(
                    name=memory["user_name"],
                    vibe=memory["ai_vibe"],
                    facts=facts_str
                )
    
    # Fallback that uses past chat context if unsure
    past_topics = [m["content"] for m in memory["messages"] if m["role"] == "user"][-3:]
    return (
        f"I'm still training my neural weights on that phrase, {memory['user_name']} "
        f"(confidence: {confidence:.0%}). Keep adding examples to `intents.json`! "
        f"By the way, I still remember our recent messages: `{past_topics}`"
    )

if prompt := st.chat_input("Talk to your custom AI..."):
    # Save & show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    memory["messages"].append({"role": "user", "content": prompt})
    save_memory(memory)

    # Generate & stream AI response smoothly
    with st.chat_message("assistant"):
        placeholder = st.empty()
        reply = generate_reply(prompt)
        typed_text = ""
        for word in reply.split(" "):
            typed_text += word + " "
            placeholder.markdown(typed_text + "▌")
            time.sleep(0.03)
        placeholder.markdown(reply)

    memory["messages"].append({"role": "assistant", "content": reply})
    save_memory(memory)
