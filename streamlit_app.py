import streamlit as st
import time
from transformers import pipeline ,  BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from huggingface_hub import login
from secret import API_TOKEN , NEW_TOKEN
import torch
#client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=API_TOKEN)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,           # 4-bit quantization
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",   # Normalized Float 4-bit
    bnb_4bit_compute_dtype=torch.float16
)
try:
    login(token="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")
    client = pipeline(
        "text2text-generation",  # T5 is a text-to-text model
        #model="google/flan-t5-small",
        #model="meta-llama/Llama-2-7b-chat-hf",
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B",gpu_layers=0),
        device=-1,
        torch_dtype=torch.float32,
        #quantization_config=bnb_config,  # Apply 4-bit

        model_kwargs={"load_in_4bit": False}# Use "cuda" if you have a GPU
    )
except Exception as e:
    st.error(f"Failed to initialize model: {str(e)}")
    st.stop()

# Strong identity enforcement
IDENTITY_RESPONSES = {
    "what is your name": "I'm Ellora AI. Think of me like an assistant who's here to help you learn, plan, and create. How can I assist you?",
    "who created you": "I was developed by Abhishek sharma with a combination of machine learning algorithms and large amounts of data . I'm constantly learning and improving, so over time I will likely become even more useful in my responses.",
    "who made you": "I was made by Abhishek Sharma with a combination of machine learning algorithms and large amounts of data .",
    "who is Abhishek sharma" : "Abhishek sharma is an AI engineer and developer and student of B.tech."
    
}
st.title("Ellora AI")
with st.sidebar:
    settings_expander = st.expander("⚙️ Settings", expanded=False)
    with settings_expander:
        max_tokens = st.slider("Response Length", 100, 1000, 400)
        temperature = st.slider("Creativity", 0.1, 1.5, 0.7)

# def get_response(prompt):
#     try:
#         # Check for identity questions first
#         template = " You are Ellora AI. You think like an AI assistant who's here to help users to learn, plan, and create. Be polite and respong in general way and solve problem step by step . You are made by Abhishek sharma who is an AI engineer and developer and student of B.tech."
        
#         lower_prompt = prompt.lower()
#         for question, answer in IDENTITY_RESPONSES.items():
#             if question in lower_prompt:
#                 return answer
        
#         # Normal response generation
#         messages = [{"role": "system", "content": template}]
#         if "messages" in st.session_state:
#             messages.extend(st.session_state.messages)
        
#         messages.append({"role": "user", "content": prompt})
        
#         response = client.chat_completion(
#             messages=messages,
#             max_tokens=600,
#             temperature=0.8,
#             stream=False  # Disable Hugging Face streaming
#         ).choices[0].message.content
        
#         return response
#     except Exception as e:
#         st.error(f"API Error: {str(e)}")
#         return "I'm having technical difficulties. Please try again later."

def get_response(prompt):
    try:
        # Check for identity questions first
        lower_prompt = prompt.lower()
        for question, answer in IDENTITY_RESPONSES.items():
            if question in lower_prompt:
                return answer

        # Normal response generation
        template = "You are Ellora AI. You think like an AI assistant who's here to help users learn, plan, and create. Be polite and respond in a general way, solving problems step by step. You were made by Abhishek Sharma, an AI engineer and developer."
        messages = [{"role": "system", "content": template}]
        if "messages" in st.session_state:
            messages.extend(st.session_state.messages)
        
        messages.append({"role": "user", "content": prompt})
        # Format the input for T5 (add the template to the prompt)
        #input_text = f"{template}\n\nUser: {prompt}\nAI:"
        
        # Generate response
        response = client(
            prompt,
            max_length=max_tokens,
            temperature=temperature,
            #stream=False,
            do_sample=True
        )[0]['generated_text']
        
        return response.split#("AI:")[-1].strip()  # Extract only the AI's part
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "I'm having technical difficulties. Please try again later."

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
    st.session_state.messages.append({"role": "assistant", "content": response})
