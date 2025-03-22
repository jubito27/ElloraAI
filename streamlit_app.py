from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import torch
import streamlit as st

# Load the model configuration and disable Sliding Window Attention
config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
config.use_sliding_window_attention = False  # Disable Sliding Window Attention

# Load the model and tokenizer

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto", low_cpu_mem_usage=True, attn_implementation="eager")
model = model.to(device)

# Create a Hugging Face pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=512,    # Maximum number of tokens to generate
    truncation=True,      # Truncate input if it exceeds max_length
    temperature=1.0,      # Adjust this value based on your use case
    return_full_text=False,  # Do not include the input prompt in the output
)

# Wrap the pipeline in LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Define the system-level prompt template
template = """You are a helpful AI assistant and your name is Ellora made by Abhishek sharma. Answer the user's questions clearly and concisely.

Conversation History:
{history}

User: {input}
Ellora:"""

# Create the PromptTemplate
prompt = PromptTemplate(input_variables=["history", "input"], template=template)

# Add memory for conversation history
memory = ConversationBufferMemory()

# Create a chatbot chain
chatbot = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False  # Set to True to see the prompt and memory in action
)

# Streamlit app
st.title("Ellora AI")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message here...")

# Handle user input
if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate chatbot response
    response = chatbot.run(user_input)
    
    # Add chatbot response to chat history
    st.session_state.chat_history.append({"role": "Ellora", "content": response})
    with st.chat_message("Ellora"):
        st.markdown(response)
