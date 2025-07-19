# 🧠 Ellora AI — Multi-Personality Assistant

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://elloraai.streamlit.app/)


Ellora AI is an intelligent, customizable AI assistant built with **Streamlit** and **Gemini**, capable of switching between personalities like:
- 😏 Sarcastic
- 😊 Friendly
- 🧑‍💼 Professional
- 🔉 Vedic Vyasa (spiritual RAG mode)
- ⚕️ Medic Expert (medical RAG mode)

It also supports:
- Voice Input 🎤
- Voice Output 🔡️
- File & Image Upload 📄🖼️
- Chat Memory (Session-based)
- Retrieval-Augmented Generation using **ChromaDB**

---

## 🔧 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ellora-ai.git
cd ellora-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r packages.txt
```

### 3. Set API Keys
Edit your environment or hardcode:
```bash
export GOOGLE_API_KEY=your_gemini_api_key
```

### 4. Run the App
```bash
streamlit run ellora_ai.py
```

---

## 📂 Project Structure
```
├── ellora_ai.py              # Main Streamlit interface
├── Ellora_vyasa.py           # RAG integration via ChromaDB
├── chromadbstore/            # Vector DB folder (created automatically)
├── requirements.txt
└── README.md
```

---

## ✨ Features

* 🤖 Gemini 1.5/2.0 API-based chat
* 🧠 Retrieval from your own PDF/Docs (RAG)
* 🎤 Voice to Text + 🔡️ Text to Speech
* 📄 Export Chat as TXT
* 📚 Persona-driven smart prompting
* 💾 Memory-aware replies (within session)

---

## 🙏 Credits

Built by **Abhishek Sharma** with help of Chatgpt and deepseek using Gemini & Langchain.

---

## 📜 License

This project is open for educational & personal use. Attribution encouraged.
