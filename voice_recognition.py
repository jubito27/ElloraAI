import streamlit as st
from streamlit.components.v1 import html
import base64
import time

def speech_to_text():
    """Browser-based speech recognition with visual feedback"""
    # Generate unique ID for this component
    component_id = f"voice_recognition_{int(time.time()*1000)}"
    
    # HTML/JS implementation
    voice_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}
            .mic-button {{
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: #ff4b4b;
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                border: none;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
            }}
            .mic-button.listening {{
                animation: pulse 1.5s infinite;
                background: #4CAF50;
            }}
            .status-text {{
                margin-top: 10px;
                text-align: center;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div style="display: flex; flex-direction: column; align-items: center;">
            <button id="micButton" class="mic-button" onclick="startListening()">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                    <line x1="12" y1="19" x2="12" y2="23"></line>
                    <line x1="8" y1="23" x2="16" y2="23"></line>
                </svg>
            </button>
            <div id="statusText" class="status-text">Click to speak</div>
        </div>

        <script>
            let recognition;
            const micButton = document.getElementById('micButton');
            const statusText = document.getElementById('statusText');
            
            // Check for speech recognition support
            if (!('webkitSpeechRecognition' in window)) {{
                statusText.textContent = "Voice input not supported in this browser";
                micButton.disabled = true;
            }} else {{
                recognition = new webkitSpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = false;
                recognition.lang = 'en-US';
                
                recognition.onstart = function() {{
                    micButton.classList.add('listening');
                    statusText.textContent = "Listening...";
                }};
                
                recognition.onresult = function(event) {{
                    const transcript = event.results[0][0].transcript;
                    window.parent.postMessage({{
                        type: 'streamlit:transcript',
                        data: transcript,
                        component_id: '{component_id}'
                    }}, '*');
                }};
                
                recognition.onerror = function(event) {{
                    console.error('Speech recognition error', event.error);
                    stopListening();
                    statusText.textContent = "Error: " + event.error;
                }};
                
                recognition.onend = function() {{
                    stopListening();
                }};
                
                function startListening() {{
                    try {{
                        recognition.start();
                    }} catch(e) {{
                        statusText.textContent = "Error: " + e.message;
                    }}
                }}
                
                function stopListening() {{
                    micButton.classList.remove('listening');
                    statusText.textContent = "Click to speak";
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    # Create component and handle results
    html(voice_html, height=120)
    
    # JavaScript to Python communication
    js_to_py = f"""
    <script>
        window.addEventListener('message', (event) => {{
            if (event.data.type === 'streamlit:transcript' && 
                event.data.component_id === '{component_id}') {{
                Streamlit.setComponentValue(event.data.data);
            }}
        }});
    </script>
    """
    html(js_to_py, height=0)
    
    # Return any captured transcript
    if st.session_state.get('voice_transcript'):
        transcript = st.session_state.voice_transcript
        del st.session_state.voice_transcript
        return transcript
    return None

def voice_input_component():
    """Streamlit component wrapper for voice input"""
    st.markdown("### Voice Input")
    with st.spinner("Initializing voice recognition..."):
        transcript = speech_to_text()
    
    if transcript:
        st.session_state.voice_transcript = transcript
        st.experimental_rerun()
    
    return transcript
