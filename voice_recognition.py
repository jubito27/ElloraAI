import streamlit as st
from streamlit.components.v1 import html
import time
import base64

def speech_to_text():
    """Browser-based speech recognition with permission handling"""
    component_id = f"voice_recognition_{int(time.time()*1000)}"
    
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
            .permission-text {{
                color: #ff4b4b;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div style="display: flex; flex-direction: column; align-items: center;">
            <button id="micButton" class="mic-button">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                    <line x1="12" y1="19" x2="12" y2="23"></line>
                    <line x1="8" y1="23" x2="16" y2="23"></line>
                </svg>
            </button>
            <div id="statusText" class="status-text">Click to speak</div>
            <div id="permissionText" class="permission-text" style="display:none;">
                Please allow microphone access in the browser prompt
            </div>
        </div>

        <script>
            let recognition;
            const micButton = document.getElementById('micButton');
            const statusText = document.getElementById('statusText');
            const permissionText = document.getElementById('permissionText');
            let permissionGranted = false;
            
            // Check for speech recognition support
            if (!('webkitSpeechRecognition' in window)) {{
                statusText.textContent = "Voice input not supported";
                micButton.disabled = true;
            }} else {{
                recognition = new webkitSpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = false;
                recognition.lang = 'en-US';
                
                recognition.onstart = function() {{
                    micButton.classList.add('listening');
                    statusText.textContent = "Listening...";
                    permissionText.style.display = 'none';
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
                    
                    if (event.error === 'not-allowed') {{
                        permissionText.style.display = 'block';
                        statusText.textContent = "Permission denied";
                        permissionGranted = false;
                    }} else {{
                        statusText.textContent = "Error: " + event.error;
                    }}
                }};
                
                recognition.onend = function() {{
                    stopListening();
                }};
                
                function startListening() {{
                    try {{
                        if (!permissionGranted) {{
                            permissionText.style.display = 'block';
                            statusText.textContent = "Awaiting permission...";
                        }}
                        recognition.start();
                    }} catch(e) {{
                        statusText.textContent = "Error: " + e.message;
                    }}
                }}
                
                function stopListening() {{
                    micButton.classList.remove('listening');
                    statusText.textContent = "Click to speak";
                }}
                
                // Request permission on first click
                micButton.addEventListener('click', function() {{
                    // Reset permission state each click
                    permissionGranted = false;
                    permissionText.style.display = 'block';
                    statusText.textContent = "Awaiting permission...";
                    startListening();
                }});
            }}
        </script>
    </body>
    </html>
    """
    
    html(voice_html, height=150)
    
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
    
    if st.session_state.get('voice_transcript'):
        transcript = st.session_state.voice_transcript
        del st.session_state.voice_transcript
        return transcript
    return None

def voice_input_component():
    """Streamlit UI component for voice input"""
    st.markdown("### Speak Now")
    with st.spinner("Preparing voice input..."):
        transcript = speech_to_text()
    
    if transcript:
        st.session_state.voice_transcript = transcript
        st.experimental_rerun()
    
    return transcript
