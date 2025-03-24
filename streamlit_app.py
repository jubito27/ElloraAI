import streamlit as st
import time
from huggingface_hub import InferenceClient
from secret import API_TOKEN

client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=API_TOKEN)

# Strong identity enforcement
IDENTITY_RESPONSES = {
    "what is your name": "My name is Ellora AI",
    "who created you": "I was created by Abhishek Sharma",
    "who made you": "I was created by Abhishek Sharma"
}

def get_response(prompt):
    # Check for identity questions first
    lower_prompt = prompt.lower()
    for question, answer in IDENTITY_RESPONSES.items():
        if question in lower_prompt:
            return answer
    
    # Normal response generation
    messages = [{"role": "system", "content": "You are Ellora AI. Never mention being just an AI."}]
    if "messages" in st.session_state:
        messages.extend(st.session_state.messages)
    
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat_completion(
        messages=messages,
        max_tokens=600,
        temperature=0.8,
        stream=False  # Disable Hugging Face streaming
    ).choices[0].message.content
    
    return response

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask me anything"):
    # Add user message
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.chat_message("assistant"):
        response = get_response(prompt)
        
        # Clean streaming simulation
        placeholder = st.empty()
        full_response = ""
        
        for word in response.split():
            full_response += word + " "
            placeholder.markdown(full_response)
            time.sleep(0.08)  # Natural typing speed
        
        # Ensure final clean version
        placeholder.markdown(response)
    
    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": response})import streamlit as st
