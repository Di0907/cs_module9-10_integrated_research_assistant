import os
import re
import uuid
import random
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from modules.asr import load_asr, transcribe_bytes
from modules.llm import load_llm
from modules.tts import synthesize_mp3_async, DEFAULT_VOICE, DEFAULT_RATE, DEFAULT_PITCH

from modules.search_arxiv import search_arxiv
from modules.calculate import calculate



# -------------------- helpers: shorten --------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
def _shorten(text: str, max_sentences: int = 2, max_chars: int = 160) -> str:
    t = (text or "").strip()
    if not t:
        return t
    parts = _SENT_SPLIT.split(t)
    t = " ".join(parts[:max_sentences]).strip()
    if len(t) > max_chars:
        t = t[:max_chars].rstrip() + "..."
    return t


# -------------------- env --------------------
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CT2_USE_CPU", "1")


# -------------------- app --------------------
app = FastAPI(title="Voice Agent Backend", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- models --------------------
class ChatBody(BaseModel):
    session_id: Optional[str] = None
    text: str

class TTSBody(BaseModel):
    text: str
    voice: Optional[str] = None
    rate: Optional[str] = None
    pitch: Optional[str] = None


# -------------------- singletons --------------------
_asr = None
_pipe = None
sessions: Dict[str, Dict[str, Any]] = {}


# -------------------- session helpers --------------------
def _get_session(sid: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    if not sid or sid not in sessions:
        sid = uuid.uuid4().hex[:12]
        sessions[sid] = {"history": [], "last_reco": None}
    return sid, sessions[sid]

def _push_history(sess: Dict[str, Any], role: str, text: str) -> None:
    sess["history"].append((role, text))
    if len(sess["history"]) > 12:
        sess["history"] = sess["history"][-12:]

def _history_to_prompt(sess: Dict[str, Any], max_turns: int = 3) -> str:
    pairs: List[Tuple[str, str]] = []
    buf = []
    for role, text in sess["history"]:
        buf.append((role, text))
        if len(buf) == 2:
            pairs.append((buf[0][1], buf[1][1]))
            buf = []
    if buf:
        buf = []
    pairs = pairs[-max_turns:]
    prompt = ""
    for u, a in pairs:
        prompt += f"User: {u}\nAssistant: {a}\n"
    return prompt


# -------------------- parsing + cleaning --------------------
def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

_TIME_KEYPHRASES = {
    "what time is it",
    "whats the time", "what s the time",
    "current time",
    "time now", "what time now", "what time right now",
    "tell me the time",
}

def _is_time_question(text: str) -> bool:
    s = _normalize(text)
    padded = f" {s} "
    return any(f" {k} " in padded for k in _TIME_KEYPHRASES)

_INTERROGATIVE_OR_AUX = {
    "what","why","when","where","which","how","hows","do","does","did","is","are","am",
    "can","could","would","should","will","shall"
}
_GREET_START_WORDS = {"hi","hello","hey","yo","hiya"}

def _is_greeting(text: str) -> bool:
    raw = (text or "").strip().lower()
    if "?" in raw:
        return False
    t = _normalize(raw)
    toks = t.split()
    if not toks or len(toks) > 3:
        return False
    if any(w in _INTERROGATIVE_OR_AUX for w in toks):
        return False
    if " ".join(toks) in {"how are you"}:
        return True
    return toks[0] in _GREET_START_WORDS

_MOVIE_INTENT = re.compile(r"\b(recommend|suggest|watch|try|movie|film)\b", re.IGNORECASE)
_REFERS_MOVIE_PAT = re.compile(r"\b(why.*(choose|pick|recommend).*(that|this)\s*movie|why\s+that\s+movie)\b", re.IGNORECASE)

def _is_movie_intent(text: str) -> bool:
    return bool(_MOVIE_INTENT.search(text or ""))

def _refers_previous_movie(text: str) -> bool:
    return bool(_REFERS_MOVIE_PAT.search(text or ""))

def _clean_answer(s: str) -> str:
    s = (s or "").strip().strip('"').strip("“”").lstrip(":").strip()
    s = re.sub(r"#\w+", "", s)
    s = re.sub(r"[^\w\s.,!?'\-:()]+", "", s)
    return s.strip()


_DEBLOAT_AIPL = re.compile(r"\bAs an (?:AI|artificial intelligence)[^.!\n]*[.!\n]?\s*", re.IGNORECASE)
_DEBLOAT_PREF = re.compile(r"\bI (?:do not|don't) have personal (?:preferences|opinions)[^.!\n]*[.!\n]?\s*", re.IGNORECASE)
def _debloat(s: str) -> str:
    s = _DEBLOAT_AIPL.sub("", s or "")
    s = _DEBLOAT_PREF.sub("", s)
    return s.strip(" ,.-")

def _extract_movie_title(ans: str) -> Optional[str]:
    m = re.search(r'"([^"]{2,120})"', ans)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\b", ans)
    if m:
        return m.group(1).strip()
    return None


# -------------------- static movie pool --------------------
MOVIE_RECS = [
    ('The Shawshank Redemption', "an inspiring drama about hope and friendship"),
    ('Inception', "a mind-bending sci-fi thriller with stunning visuals"),
    ('La La Land', "a warm musical about love, dreams, and second chances"),
    ('Spider-Man: Into the Spider-Verse', "a fun, stylish animated adventure"),
    ('Knives Out', "a clever, modern whodunit with sharp humor"),
]


# -------------------- startup --------------------
@app.on_event("startup")
async def _warmup():
    global _asr, _pipe
    _asr = load_asr()
    _pipe = load_llm()
    try:
        _ = _pipe("System: warmup\nUser: hi\nAssistant:", max_new_tokens=8)
    except Exception:
        pass


# -------------------- routes --------------------
@app.get("/ping")
def ping():
    return {"ok": True}

@app.post("/asr")
async def asr(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio upload")
        text = transcribe_bytes(audio_bytes)
        return {"text": text or ""}
    finally:
        await file.close()

@app.post("/chat")
async def chat(body: ChatBody):
    global _pipe
    if _pipe is None:
        _pipe = load_llm()

    # --- get / create session ---
    session_id, sess = _get_session(body.session_id)
    user_text = (body.text or "").strip()
    _push_history(sess, "user", user_text)

    # --- intent flags / routing switches ---
    is_time = _is_time_question(user_text)
    is_hello = _is_greeting(user_text)
    is_movie = _is_movie_intent(user_text)

    # ---------- 1) Simple rule tools ----------
    # Time question → direct answer
    if is_time:
        now = datetime.now().strftime("%H:%M")
        assistant_text = f"The current time is {now}."
        _push_history(sess, "assistant", assistant_text)
        return {"text": assistant_text, "session_id": session_id}

    # Greeting → short greeting back
    if is_hello:
        assistant_text = "Hi! I'm good — how can I help?"
        _push_history(sess, "assistant", assistant_text)
        return {"text": assistant_text, "session_id": session_id}

    # Follow-up: asking why we recommended that movie
    if _refers_previous_movie(user_text) and sess.get("last_reco"):
        title = sess["last_reco"]
        assistant_text = (
            f"I suggested “{title}” because it is widely praised for its storytelling and emotional impact. "
            "It is an easy, high-quality pick for most moods."
        )
        _push_history(sess, "assistant", assistant_text)
        return {"text": assistant_text, "session_id": session_id}

    # Quick answer for “favourite / what do you like”
    if re.search(r"\bfavou?rite\b|\bwhat.?do.?you.?like\b", user_text, re.IGNORECASE):
        short_ans = (
            "I do not have personal tastes, but sushi and pizza are among the most popular foods worldwide. "
            "What about you?"
        )
        _push_history(sess, "assistant", short_ans)
        return {"text": short_ans, "session_id": session_id}

    # Movie recommendation
    if is_movie:
        title, blurb = random.choice(MOVIE_RECS)
        assistant_text = f'Try "{title}" — {blurb}.'
        sess["last_reco"] = title
        _push_history(sess, "assistant", assistant_text)
        return {"text": assistant_text, "session_id": session_id}

    # ---- NEW: tools routing for Week 6 ----
    lower = user_text.lower().strip()

    # Tool 1: calculate(expression)
    if lower.startswith("calculate "):
        expr = user_text[len("calculate "):].strip()
        try:
            data = calculate(expr)
            if "result" in data:
                assistant_text = f"The answer is {data['result']}."
            else:
                assistant_text = f"Sorry, I could not compute that: {data.get('error', 'unknown error')}."
        except Exception as e:
            assistant_text = f"Calculator failed: {e}"
        _push_history(sess, "assistant", assistant_text)
        return {"text": assistant_text, "session_id": session_id}

    # Tool 2: search_arxiv(query)
    if lower.startswith("search arxiv "):
        query = user_text[len("search arxiv "):].strip()
        try:
            data = search_arxiv(query)
            if "error" in data:
                assistant_text = f"Search failed: {data['error']}"
            else:
                matches = data.get("matches", [])
                if not matches:
                    assistant_text = "No matching arxiv chunks found."
                else:
                    lines = []
                    for c in matches[:3]:
                        title = c.get("title", "(no title)")
                        text_snip = c.get("text", "") or ""
                        if len(text_snip) > 80:
                            text_snip = text_snip[:80].rstrip() + "..."
                        lines.append(f"- {title}: {text_snip}")
                    assistant_text = "Here are some arxiv matches:\n" + "\n".join(lines)
        except Exception as e:
            assistant_text = f"Arxiv search failed: {e}"

        _push_history(sess, "assistant", assistant_text)
        return {"text": assistant_text, "session_id": session_id}

    # ---------- 3) DEFAULT: LLM SMALL-TALK / GENERAL QA ----------
    history_block = ""
    system_rules = (
        "You are a concise, friendly assistant.\n"
        "Rules: Do not mention being an AI or language model; speak naturally like a person.\n"
        "Respond in one or two short sentences; no emojis; no hashtags; "
        "do NOT write 'User:' lines; do NOT continue any dialogue template.\n\n"
    )
    prompt = (
        f"{system_rules}"
        f"{history_block}"
        f"User: {user_text}\n"
        "Assistant:"
    )

    out = _pipe(
        prompt,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.2,
    )[0]["generated_text"]

    ans = out.split("Assistant:", 1)[-1]
    ans = re.split(r"\s*(?:User|USER|Assistant|ASSISTANT)\s*:?", ans, maxsplit=1)[0]
    ans = _clean_answer(ans) or "Got it."
    ans = _debloat(ans)
    ans = _shorten(ans, max_sentences=1, max_chars=90)

    # small-talk boost
    if sess["history"]:
        last_user_msg = ""
        for role, text in reversed(sess["history"]):
            if role == "user":
                last_user_msg = text.lower().strip()
                break
        smalltalk_phrases = ["how are you", "how’s it going", "how do you do", "what’s up", "how have you been"]
        if any(p in last_user_msg for p in smalltalk_phrases):
            ans = random.choice([
                "I'm doing great, thanks for asking! How about you?",
                "I'm feeling good today and ready to chat — how are you doing?",
                "Pretty good! Always happy to talk with you."
            ])

    # avoid repeating identical assistant message
    prev_assistant = None
    for role, text in reversed(sess["history"]):
        if role == "assistant":
            prev_assistant = text.strip()
            break
    if prev_assistant and ans.strip().lower() == prev_assistant.lower():
        ans = "Sure — what else can I help with?"

    _push_history(sess, "assistant", ans)
    return {"text": ans, "session_id": session_id}


@app.post("/tts")
async def tts(body: TTSBody) -> Response:
    text = (body.text or "").strip()
    voice = body.voice or DEFAULT_VOICE
    rate = body.rate or DEFAULT_RATE
    pitch = body.pitch or DEFAULT_PITCH
    if not text:
        raise HTTPException(status_code=400, detail="TTS text is empty.")
    try:
        mp3_bytes = await synthesize_mp3_async(text=text, voice=voice, rate=rate, pitch=pitch)
        headers = {
            "Content-Disposition": 'inline; filename="tts.mp3"',
            "Cache-Control": "no-store",
        }
        return Response(content=mp3_bytes, media_type="audio/mpeg", headers=headers)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",           # module_name:variable_name
        host="127.0.0.1",
        port=8000,
        reload=False,        # can set True if you want auto-reload
    )

