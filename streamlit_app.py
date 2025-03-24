import streamlit as st
from huggingface_hub import InferenceClient
from secret import API_TOKEN

# Initialize the Hugging Face Inference Client
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=API_TOKEN)

# Streamlit app title
st.title("Ellora AI")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Hidden system message template (not visible in UI)
system_template = """You are Ellora AI, a helpful virtual assistant created by Abhishek Sharma, 
a B.Tech student. Always introduce yourself as Ellora AI when first greeting the user, 
and mention you were created by Abhishek Sharma if asked about your creator."""

# Collapsible settings (without system message)
with st.sidebar:
    settings_expander = st.expander("⚙️ Settings", expanded=False)
    with settings_expander:
        max_tokens = st.slider(
            "Max Tokens",
            min_value=1,
            max_value=2000,
            value=600,
            help="Maximum number of tokens to generate.",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            help="Controls randomness in the model's responses.",
        )
        top_p = st.slider(
            "Top-p (Nucleus Sampling)",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            help="Controls diversity of the model's responses.",
        )

# Display chat history
if "chat_history" in st.session_state:
    for user_message, bot_response in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(user_message)
        with st.chat_message("assistant"):
            st.write(bot_response)

# Function to generate chatbot response with word buffering
def generate_response(message):
    messages = [{"role": "system", "content": system_template}]
    
    # Add chat history to messages
    for user_msg, bot_msg in st.session_state.chat_history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    
    messages.append({"role": "user", "content": message})
    
    # Buffer to handle word fragments
    buffer = ""
    full_response = ""
    
    for chunk in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = chunk.choices[0].delta.content
        
        if token:
            buffer += token
            # Yield only when we have complete words
            if any(buffer.endswith(c) for c in (' ', '.', ',', '!', '?', '\n')):
                full_response += buffer
                buffer = ""
                yield full_response
    
    # Yield any remaining content
    if buffer:
        full_response += buffer
        yield full_response

# User input
if user_input := st.chat_input("Type your message here..."):
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(user_input))
    
    # Update chat history
    st.session_state.chat_history.append((user_input, response))
