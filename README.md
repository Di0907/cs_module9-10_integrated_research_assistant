# Integrated Research Assistant System — Voice Agent  
**Author:** Di Han  
**Email:** dihan9728@gmail.com  

---

## Description — What This Homework Is About

This assignment implements an **Integrated Research Assistant System** in the form of a **Voice Agent**, combining speech input, tool-augmented reasoning, retrieval, and session-level summarization.

The system is designed to demonstrate:
- multimodal interaction (voice + text)
- tool calling
- retrieval-augmented responses
- session-aware summarization
- a complete runnable backend + frontend demo

---

## System Overview

The Voice Agent supports:

- 🎙️ **ASR (Speech-to-Text)** for voice input  
- 💬 **Text-based interaction** as a reliable fallback  
- 🧠 **LLM-based reasoning** with short-term memory  
- 🔧 **Tool calling**
  - calculator
  - arXiv-style academic search
- 📚 **Retrieval** over provided academic chunks  
- 📝 **Session-level summary generation** (triggered by *End Session*)  
- 🌐 **FastAPI backend + browser-based frontend**
- ⚙️ **One-click startup using a Windows `.bat` script**

---

## Features Implemented

### 1. ASR (Speech → Text)
**File:** `modules/asr.py`  
Converts microphone input into text for downstream processing.

---

### 2. Calculator Tool
**File:** `modules/calculate.py`  
Safely evaluates mathematical expressions and returns computed results.

---

### 3. Academic Search Tool (arXiv-style)
**File:** `modules/search_arxiv.py`  

- Searches over provided academic text chunks
- Returns ranked text fragments
- Retrieval results are logged into the session memory and reused in summaries

---

### 4. Session-Aware Conversation Management
**File:** `modules/session_manager.py`  

Tracks:
- user turns
- assistant turns
- retrieval hits
- session statistics  

Supports a clean **End Session** action that:
- stops the conversation
- generates a structured session summary instantly
- resets the session state for the next interaction

---

### 5. Fast Session Summary (No LLM Call)
**File:** `app.py`  

When the user types **“end session”** or clicks **End Session**, the system produces:

- TITLE  
- SUMMARY bullets  
- FOLLOW-UP QUESTIONS  
- KEY EVIDENCE (from retrieved documents)  

This summary is generated **instantly**, without calling the LLM, ensuring fast and reliable behavior for demo purposes.

---

### 6. Frontend Demo Interface
**Folder:** `client/`

- Text input + microphone support
- Send button + End Session button
- Displays conversation transcript and session summary
- Designed for screen-recorded demo

---

## Project Structure
```text
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
│   └── session_manager.py    # Session tracking + "End Session" summary
├── client/
│   ├── index.html            # UI: input box + mic button + Send / End Session payload builder
│   └── client.js             # Browser logic: record mic, call /asr and /chat, render logs
└── .gitignore
```
---

## How to Run the System

### Option 1 — One-click (Recommended)
Double-click:
start_voice_agent.bat


This will:
1. Start the FastAPI backend on port `8000`
2. Start the frontend static server on port `8080`
3. Automatically open the browser demo page

---

### Option 2 — Manual Startup

Install dependencies:
pip install -r requirements.txt

Start backend:
python app.py

Open frontend:
http://127.0.0.1:8080/client/index.html

---

### Key API Endpoints
GET /ping
Health check.

POST /asr
Uploads audio and returns recognized text.

POST /chat
Main interaction endpoint:
-routes user input
-triggers tools when needed
-records session history
-generates session summary on End Session

---

### Example Interaction Flow (Demo)
1.User speaks or types:
search arxiv transformers
2.System returns relevant academic matches
3.User clicks End Session
4.System generates:
-session title
-concise summary
-follow-up questions
-key evidence from retrieved papers

---

### Excluded / Ignored Files
The following are excluded or safe to delete:
-__pycache__/
-temporary TTS audio files
-local logs
All exclusions are handled via .gitignore.

---

### Submission Notes
-Demo video uploaded to course platform
-GitHub repository contains full runnable source code
-System runs locally without external services
-All required components demonstrated
-Session summary functionality implemented as required


