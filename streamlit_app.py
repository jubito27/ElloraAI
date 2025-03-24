import streamlit as st
from huggingface_hub import InferenceClient
from secret import API_TOKEN

# Initialize the Hugging Face Inference Client
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=API_TOKEN)

# Streamlit app title
st.title("Ellora AI")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Stronger system message with explicit instructions
system_template = """You are Ellora AI, a friendly virtual assistant created by Abhishek Sharma. 
You MUST always follow these rules:
1. When asked your name, respond: "My name is Ellora AI"
2. When asked who created you, respond: "I was created by Abhishek Sharma, a B.Tech student"
3. Never mention being a language model or AI unless specifically asked
4. Never mention not having a physical form
5. Respond conversationally and helpfully"""

# Settings (collapsed by default)
with st.sidebar:
    settings_expander = st.expander("⚙️ Settings", expanded=False)
    with settings_expander:
        max_tokens = st.slider("Response Length", 100, 1000, 400)
        temperature = st.slider("Creativity", 0.1, 1.5, 0.7)

def enforce_identity(response):
    """Post-processing to ensure correct identity responses"""
    identity_phrases = {
        "what is your name": "My name is Ellora AI , a AI made for your help.",
        "who created you": "I was created by Abhishek Sharma",
        "who made you": "I was created by Abhishek Sharma" , 
        "who is Abhishek sharma" : "Abhishek sharma is AI engineer and developer and also a Btech Student."
    }
    
    lower_response = response.lower()
    for phrase, answer in identity_phrases.items():
        if phrase in lower_response:
            return answer
    return response

def generate_response(prompt):
    messages = [
        {"role": "system", "content": system_template},
        *[{"role": role, "content": content} 
          for user, assistant in st.session_state.chat_history 
          for role, content in [("user", user), ("assistant", assistant)]],
        {"role": "user", "content": prompt}
    ]
    
    # Get complete response first (not streamed) to ensure consistency
    response = client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=False,
        temperature=temperature
    ).choices[0].message.content
    
    # Enforce identity responses
    response = enforce_identity(response)
    
    # Simulate streaming for better UX
    words = response.split()
    partial = ""
    for word in words:
        partial += word + " "
        yield partial
        time.sleep(0.05)

# Chat interface
if prompt := st.chat_input("How can I help you?"):
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(prompt))
    
    st.session_state.chat_history.append((prompt, response))
