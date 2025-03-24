import streamlit as st
from huggingface_hub import InferenceClient
from secret import API_TOKEN

client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=API_TOKEN)

st.title("Ellora AI")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Hidden system prompt
template = """You are Ellora AI, a helpful virtual assistant created by Abhishek Sharma, a B.Tech student. 
Respond concisely and helpfully to user queries and do not repeat the same lines again and again , reply like you high IQ genius AI ."""

# Collapsible settings
with st.sidebar:
    settings_expander = st.expander("⚙️ Settings", expanded=False)
    with settings_expander:
        max_tokens = st.slider("Max Tokens", 50, 1000, 400)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.9)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.95)

# Display chat history
for user, assistant in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(user)
    with st.chat_message("assistant"):
        st.write(assistant)

def generate_response(prompt):
    template = """You are Ellora AI, a helpful virtual assistant created by Abhishek Sharma, a B.Tech student. 
                    Respond concisely and helpfully to user queries and do not repeat the same lines again and again , reply like you high IQ genius AI ."""

    messages = [{"role": "system", "content": template}]
    
    # Add conversation history
    for user_msg, bot_msg in st.session_state.chat_history:
        messages.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": bot_msg}
        ])
    
    messages.append({"role": "user", "content": prompt})
    
    # Buffer to hold complete words
    buffer = ""
    full_response = ""
    
    for chunk in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p
    ):
        token = chunk.choices[0].delta.content
        
        if token:
            buffer += token
            
            # Only yield when we have complete words followed by space/punctuation
            if buffer.endswith((' ', '.', ',', '!', '?', '\n')):
                full_response += buffer
                buffer = ""
                yield full_response

    # Yield any remaining content in buffer
    if buffer:
        full_response += buffer
        yield full_response

# Chat input
if prompt := st.chat_input("Type your message..."):
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(prompt))
    
    st.session_state.chat_history.append((prompt, response))
