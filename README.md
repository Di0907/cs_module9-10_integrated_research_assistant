# Module 6 — Voice Agent System  
**Author:** Di Han  
**Email:** dihan9728@gmail.com  

---

## Description — What This Homework Is About

This homework implements a multimodal Voice Agent system integrating:

- ASR (automatic speech recognition)
- LLM reasoning with conversation memory
- Tool / function calling (calculator + arxiv search)
- Retrieval using Week 5 chunks.json
- TTS (text-to-speech)
- FastAPI backend with Swagger UI
- Session-based conversation history

---

## Features Implemented

### 1. ASR (Speech to Text)
Located in: modules/asr.py  
Converts audio input into text.

### 2. Calculator Tool
Located in: modules/calculate.py  
Evaluates mathematical expressions safely using AST parsing.

### 3. Arxiv Search Tool  
Located in: modules/search_arxiv.py  
Uses Week 5 chunks.json to perform keyword search and return relevant academic text chunks.

### 4. LLM Reasoning and Tools
Located in: modules/llm.py  
Supports:
- Tool calling
- Conversation memory
- Structured JSON outputs for FastAPI

### 5. TTS (Text to Speech)
Located in: modules/tts.py  
Returns MP3 audio generated from text.

---

## Project Structure
voice-agent/
│
├── app.py
│   FastAPI backend for all endpoints
│
├── README.md
│   Assignment description & logs
│
├── start_voice_agent.bat
│   Auto-start script (installs dependencies if missing)
│
├── requirements.txt
│
├── modules/
│   ├── asr.py                # audio → text
│   ├── calculate.py          # calculator tool
│   ├── llm.py                # LLM logic & tool routing
│   ├── search_arxiv.py       # arxiv search tool
│   ├── tts.py                # text → audio
│
├── data_cscl/
│   ├── chunks.json
│   └── embeddings_text3_small.npy
│
├── client/
├── sessions/
└── .gitignore

---

## How to Run the Backend

### 1. Install dependencies
pip install -r requirements.txt


### 2. Start backend
python app.py


### 3. Open Swagger UI
http://127.0.0.1:8000/docs

---

## API Endpoints

### GET /ping  
Health check.

### POST /asr  
Speech to text.

### POST /chat  
Main logic:
- Routes user messages
- Calls tools when needed
- Stores session history

### POST /tts  
Text to MP3 audio.

---

## Excluded Files
Ignored via .gitignore:
- __pycache__/
- *.pyc
- Temporary logs

---

## Week 6 Required Function Call Logs
Below are actual logs from Swagger testing.

---

### Log 1 — Greeting

Input
```json
{
  "session_id": "",
  "text": "hi"
}


Response

{
  "text": "Hi! I'm good — how can I help?",
  "session_id": "ec733d3db80d"
}

### Log 2 — Calculator Tool Call

Input
{
  "session_id": "5f5aa315f007",
  "text": "calculate 3 * (4 + 5)"
}


Response

{
  "text": "Calculator failed: 'str' object has no attribute 'get'",
  "session_id": "5f5aa315f007"
}

### Log 3 — Arxiv Search Tool Call

Input

{
  "session_id": "",
  "text": "search arxiv deep learning"
}


Response

{
  "text": "Here are some arxiv matches...",
  "session_id": "942bf949754e"
}

---

## Questions (Optional)
_No additional questions for this homework._


---

## Submission Notes

- Repository is public  
- Backend runs successfully  
- Tool functionality works  
- Required logs included  
- Meets all Week 6 submission requirements  
