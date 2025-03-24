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

# System details (hidden from the user)
template = "A helpful AI assistant named Ellora AI created by Abhishek Sharma, a B.Tech student."

# Sidebar for settings (collapsible)
with st.sidebar:
    st.header("Settings")
    # Add a checkbox to toggle settings visibility
    show_settings = st.checkbox("Show Settings")

    if show_settings:
        system_message = st.text_area(
            "System Details",
            value=template,
            help="Define the system-level behavior of the chatbot.",
        )
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
    else:
        # Use default values if settings are not shown
        system_message = template
        max_tokens = 600
        temperature = 1.0
        top_p = 0.9

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
        if token:  # Ensure token is not None or empty
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
