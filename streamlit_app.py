import streamlit as st
from huggingface_hub import InferenceClient

# Initialize the Hugging Face Inference Client
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Streamlit app title
st.title("Chat with Zephyr-7b-beta")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for system message and parameters
with st.sidebar:
    st.header("Settings")
    system_message = st.text_area(
        "System Message",
        value="You are a helpful AI assistant.",
        help="Define the system-level behavior of the chatbot.",
    )
    max_tokens = st.slider(
        "Max Tokens",
        min_value=1,
        max_value=500,
        value=200,
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
else:
    st.warning("Chat history is not initialized. Please refresh the page.")

# Function to generate chatbot response
def respond(message, system_message, max_tokens, temperature, top_p):
    messages = [{"role": "system", "content": system_message}]

    # Add chat history to messages
    if "chat_history" in st.session_state:
        for user_msg, bot_msg in st.session_state.chat_history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

    # Add the current user message
    messages.append({"role": "user", "content": message})

    # Stream the response from the model
    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Ensure chat history is initialized
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Generate and display bot response
    with st.chat_message("assistant"):
        bot_response = st.write_stream(
            respond(user_input, system_message, max_tokens, temperature, top_p)
        )

    # Update chat history
    st.session_state.chat_history.append((user_input, bot_response))
