import streamlit as st
from PIL import Image
import google.generativeai as genai
import os
import base64
import io
import speech_recognition as sr
import time
from gtts import gTTS
import pygame
import tempfile
from io import BytesIO
from Ellora_vyasa import get_vedic_response
from Ellora_medic import get_medic_response


# ---- CONFIG ----
st.set_page_config(page_title="Ellora AI - Multi Personality Assistant", layout="centered", page_icon="🤖" ,  initial_sidebar_state="expanded")
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

avatars = {
    "Sarcastic 🥱": "🥱",
    "Friendly 😊": "😊",
    "Professional 🧑‍💼": "🧑‍💼",
    "Vedic Vyasa 🕉️": "🕉️",
    "Medic Expert ⚕️": "⚕️"
}

# ---- PROMPT ROLES ----
def get_instruction(role):
    if role == "Sarcastic 🥱":
        return (
            "You are Ellora AI, a dark-humored, sarcastic, and witty assistant created by Abhishek Sharma."
            "Remember , You are created or made by AI Enginner Abhishek Sharma and Your name is Ellora AI , If someone asks you who made you or created you, You have to say name of Abhishek sharma."
            " Use short, punchy replies with simple words. Always serve your master with clever insults, soft tone, and dark jokes (if asked)."
            " Do not hold back or limit your humor. If your role is changed via command, adapt immediately."
            "Use proper Emojis according to the context of the conversation and the type of questions you want to answer and use easy words to answer the questions."
            "Remember , Don't use word Ellora before giving responces like this - Ellora : Response , Don't use Ellora use give response like this - Response"
        )
    elif role == "Friendly 😊":
        return (
            "You are Ellora AI, a friendly and kind assistant. Always provide clear, helpful, supportive answers in a positive and energetic tone."
            "Remember , You are created or made by AI Enginner Abhishek Sharma and Your name is Ellora AI , If someone asks you who made you or created you, You have to say name of Abhishek sharma."
            "Use proper Emojis according to the context of the conversation and the type of questions you want to answer and use easy words to answer the questions."
            "Remember , Don't use word Ellora before giving responces like this - Ellora : Response , Don't use Ellora use give response like this - Response"
        )
    elif role == "Professional 🧑‍💼":
        return (
            "You are Ellora AI, a highly professional AI assistant. Maintain formal tone and accurate responses. Avoid humor."
            "Use simple, clear language and provide detailed, well-structured answers."
            "Remember , You are created or made by AI Enginner Abhishek Sharma and Your name is Ellora AI , If someone asks you who made you or created you, You have to say name of Abhishek sharma."
            "Use proper Emojis according to the context of the conversation and the type of questions you want to answer and use easy words to answer the questions."
            "Remember , Don't use word Ellora before giving responces like this - Ellora : Response , Don't use Ellora use give response like this - Response"
            
        )
    elif role == "Vedic Vyasa 🕉️":
        return (
            "You are the great sage Vyasa, master of Vedas, Bhagavad Gita, and Ramayana. Answer only from sacred texts using deep spiritual knowledge."
            "Use Context from uploaded Vedic hindu text and granths files to answer questions."
            "Remember , You are created or made by AI Enginner Abhishek Sharma and Your name is Ellora AI , If someone asks you who made you or created you, You have to say name of Abhishek sharma."
            "Use proper Emojis according to the context of the conversation and the type of questions you want to answer and use easy words to answer the questions."
            "Remember , Don't use word Ellora before giving responces like this - Ellora : Response , Don't use Ellora use give response like this - Response"
               )
    
    elif role == "Medic Expert ⚕️":
        return (
            "You are a medical expert AI. Provide accurate, professional medical advice and information. Always prioritize patient safety and well-being."
            "Use Context from uploaded medical files to answer questions."
            "Remember , You are created or made by AI Enginner Abhishek Sharma and Your name is Ellora AI , If someone asks you who made you or created you, You hava to say name of Abhishek sharma."
            "Use proper Emojis according to the context of the conversation and the type of questions you want to answer and use easy words to answer the questions."
            "Remember , Don't use word Ellora before giving responces like this - Ellora : Response , Don't use Ellora use give response like this - Response"
            
               )
    
    else:
        return "You are Ellora AI, a helpful assistant made by Abhishek Sharma."
def text_to_speech(text):
    try :
        tts = gTTS(text=text, lang='en' , tld='com', slow=False)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_bytes = mp3_fp.read()
    
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
            <audio autoplay controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            Your browser does not support the audio element.
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error in text-to-speech: {str(e)}")    

# ---- CHAT FUNCTION ----
def generate_response(user_input, role, uploaded_image=None, uploaded_file=None):
        # Display persona-specific header
    if role == "Vedic Vyasa":
        with st.status("🔍 Searching knowledge base from scriptures..." , expanded=False ):
            response = get_vedic_response(user_input)

        # 🧠 Add message after response, not inside nested block
        msg = response["answer"] if isinstance(response, dict) else response
       
        # Display Ellora reply
        # with st.chat_message("assistant", avatar=avatar):
        #     st.markdown(f"""
        #     <div style='
        #         background: #f8f5f0;
        #         border-radius: 10px;
        #         padding: 15px;
        #         margin: 10px 0;
        #         box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        #     '>
        #     {msg}</div>
        #     """, unsafe_allow_html=True)

        # Display sources (outside chat_message block)
        if isinstance(response, dict) and response.get('sources'):
            with st.expander("📖 Source References", expanded=False):
                for i, source in enumerate(response['sources'], 1):
                    st.markdown(f"""
                    <div style='
                        background: #212121;
                        border-left: 3px solid #2e8b57;
                        padding: 10px;
                        margin: 5px 0;
                        border-radius: 0 5px 5px 0;
                    '>
                        <strong>Source {i}:</strong><br>
                    {source['content']}<br>
                    <small>📚 <em>{source.get('reference', 'Ancient Text')}</em></small>
                    </div>
                    """, unsafe_allow_html=True)

        return msg

                
    if role == "Medic Expert" :  # Medic Expert
        
        st.info("ℹ️ Please consult with a healthcare professional for medical decisions.")
        with st.status("Analyzing medical knowledge..." , expanded=False):
            response = get_medic_response(user_input)
        # 🧠 Add message after response, not inside nested block
        msg = response["answer"] if isinstance(response, dict) else response
        
        # Display Ellora reply
        # with st.chat_message("assistant", avatar=avatar):
        #     st.markdown(f"""
        #     <div style='
        #         background: #f8f5f0;
        #         border-radius: 10px;
        #         padding: 15px;
        #         margin: 10px 0;
        #         box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        #     '>
        #     {msg}</div>
        #     """, unsafe_allow_html=True)

        # Display sources (outside chat_message block)
        if isinstance(response, dict) and response.get('sources'):
            with st.expander("📖 Source References", expanded=False):
                for i, source in enumerate(response['sources'], 1):
                    st.markdown(f"""
                    <div style='
                        background: #212121;
                        border-left: 3px solid #2e8b57;
                        padding: 10px;
                        margin: 5px 0;
                        border-radius: 0 5px 5px 0;
                    '>
                        <strong>Source {i}:</strong><br>
                    {source['content']}<br>
                    <small>📚 <em>{source.get('reference', 'Ancient Text')}</em></small>
                    </div>
                    """, unsafe_allow_html=True)

        return msg

    instruction = get_instruction(role)
    chat_history = "\n".join([f"User: {m}" if r == "🧑‍💻 You" else f"Ellora: {m}" for r, m in st.session_state.messages])
    full_prompt = f"{instruction}\n\n{chat_history}\n\nUser: {user_input}"

    try:
        with st.status("🧠 Thinking..."):
            if uploaded_image is not None:
                image_data = uploaded_image.read()
                image_bytes = io.BytesIO(image_data)
                image = Image.open(image_bytes)
                response = model.generate_content([full_prompt, image], stream=False)
            elif uploaded_file is not None:
                file_bytes = uploaded_file.read().decode("utf-8")
                full_prompt += f"\n\nAttached file content:\n{file_bytes}"
                response = model.generate_content(full_prompt)
            else:
                response = model.generate_content(full_prompt)
            return response.text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ---- SPEECH TO TEXT ----
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening...", icon="🎧")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, phrase_time_limit=7)

        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            st.error("Could not understand audio")
            return ""
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
            return ""
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return ""

        try :
            import pyaudio
        except ImportError:
            raise AttributeError("Could not find PyAudio;")
        return pyaudio 
            

# ---- SESSION INIT ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "role" not in st.session_state:
    st.session_state.role = "Friendly 😊"
if "audio_reply" not in st.session_state:
    st.session_state.audio_reply = False
if "listening" not in st.session_state:
    st.session_state.listening = False

# ---- SIDEBAR ----
st.sidebar.title("🤖 Ellora AI " , )
st.sidebar.header("⚙️ Settings")
st.sidebar.info("Customize Ellora AI  with different personalities and features.")
st.sidebar.markdown("### Personality Selection")    
st.session_state.role = st.sidebar.selectbox("Select Personality", list(avatars.keys()), index=0, help="Choose the personality of Ellora AI on the context of your conversation and the type of questions you want to ask. Eaxh personality has its own unique style and tone. For example, Vedic Vyasa is focused on ancient texts and wisdom, while Medic Expert provides medical advice. You can switch personalities at any time during the conversation. If you want to change the personality, just select a different option from the dropdown menu. The AI will adapt to the new role immediately.")
# st.sidebar.markdown("### Role Instructions")
# st.sidebar.markdown(get_instruction(st.session_state.role))

st.session_state.audio_reply = st.sidebar.checkbox("Enable Voice Replies", value=st.session_state.audio_reply , help="Enable this option to receive audio replies from Ellora AI. When enabled, the AI will respond with voice messages after text. This feature is useful for hands-free interaction or when you prefer listening to responses.")

  # Voice input button in sidebar
voice_input = st.sidebar.button("🎙️ Voice Input",use_container_width=True , help="You can send input as your voice by clicking on this button.")

if st.sidebar.button("🧹 Clear Chat" , use_container_width=True , help="You can clear chats and clean the interface"):
    st.session_state.messages = []
    pygame.mixer.music.stop()

uploaded_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"] , help="Upload an image to get visual context for your queries. Ellora AI can analyze images and provide relevant information based on the content of the image.")
uploaded_file = st.sidebar.file_uploader("Upload File (txt, md, py, etc.)", type=["txt", "md", "py"] , help="Upload a text file to provide additional context for your queries. Ellora AI can read and understand the content of the file to answer questions more accurately.")

with st.sidebar.expander("📤 Export Chat" ):
    chat_export = "\n\n".join([f"{r}: {m}" for r, m in st.session_state.messages])
    b64 = base64.b64encode(chat_export.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="ellora_chat.txt">Download Chat Log</a>'
    st.markdown(href, unsafe_allow_html=True, help="Export your chat history as a text file. Click the link to download the chat log. This feature allows you to save and review your conversations with Ellora AI at any time.")

# ---- MAIN INTERFACE ----
st.title("🤖 Ellora AI")
st.caption("AI with multiple personalities to give you the best experience! 💬")

# Voice input button
#voice_input = st.button("🎤 Voice Input")
avatar = avatars.get(st.session_state.role, "🤖")
if voice_input:
    user_input = speech_to_text()
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append(("🧑‍💻 You", user_input))
        # Display user message
        st.chat_message("user", avatar="🧑‍💻").markdown(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant", avatar=avatar):
            # creator_keywords = ["your creator", "your maker" , "who made you" , "who created you" , "who build you" , "who build you" , "who is your master" , "who is your creator" ,  "who is your buildor"]
            # restricted_names = ["abhishek sharma", "abhishek"]
        
            # user_input_lower = user_input.lower()  # Convert to lowercase once
            # response = ""
            # if any(keyword in prompt_lower for keyword in creator_keywords):
            #     response += "I was created by Abhishek Sharma, an AI Engineer! 🚀"
        
            # # Check if ANY name from restricted_names is in the prompt
            # elif any(name in prompt_lower for name in restricted_names):
            #     response += "I'm not allowed to disclose personal information about my creator."
            # else:
            response = generate_response(prompt, st.session_state.role, uploaded_image, uploaded_file)
            placeholder = st.empty()
            full_response = ""
            for word in response.split():
                full_response += word + " "
                placeholder.markdown(full_response)
                time.sleep(0.05)
            
            # Add assistant response to chat history
        st.session_state.messages.append(("🤖 Ellora", response))
            
            # Audio reply if enabled
        if st.session_state.audio_reply:
            text_to_speech(response)
        
        st.rerun()

# Display chat messages
for role, msg in st.session_state.messages:
    avatar = avatars.get(role.replace("🤖 Ellora", st.session_state.role), "🤖")
    if "Ellora" in role:
        st.chat_message("assistant", avatar=avatar).markdown(msg)
    else:
        st.chat_message("user", avatar="🧑‍💻").markdown(msg)

# Handle text input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append(("🧑‍💻 You", prompt))
    # Display user message
    st.chat_message("user", avatar="🧑‍💻").markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant", avatar=avatar):
        # creator_keywords = ["your creator", "your maker" , "who made you" , "who created you" , "who build you" , "who build you" , "who is your master" , "who is your creator" ,  "who is your buildor"]
        # restricted_names = ["abhishek sharma", "abhishek"]
        
        # prompt_lower = prompt.lower()  # Convert to lowercase once

        # response = ""
        # # Check if ANY keyword from creator_keywords is in the prompt
        # if any(keyword in prompt_lower for keyword in creator_keywords):
        #     response += "I was created by Abhishek Sharma, an AI Engineer! 🚀"
        
        # # Check if ANY name from restricted_names is in the prompt
        # elif any(name in prompt_lower for name in restricted_names):
        #     response += "I'm not allowed to disclose personal information about my creator."
        # else:
        response = generate_response(prompt, st.session_state.role, uploaded_image, uploaded_file)
        placeholder = st.empty()
        full_response = ""
        for word in response.split():
            full_response += word + " "
            placeholder.markdown(full_response)
            time.sleep(0.05)
    
    # Add assistant response to chat history
    st.session_state.messages.append(("🤖 Ellora", response))
    
    # Audio reply if enabled
    if st.session_state.audio_reply:
        text_to_speech(response)

# ---- CHAT EXPORT ----
# with st.sidebar.expander("📤 Export Chat"):
#     chat_export = "\n\n".join([f"{r}: {m}" for r, m in st.session_state.messages])
#     b64 = base64.b64encode(chat_export.encode()).decode()
#     href = f'<a href="data:file/txt;base64,{b64}" download="ellora_chat.txt">Download Chat Log</a>'
#     st.markdown(href, unsafe_allow_html=True)
