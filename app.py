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
from modules.session_manager import SessionManager


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
app = FastAPI(title="Voice Agent Backend", version="2.1-fast-summary")
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

# sessions store:
# { sid: {"history": [(role,text),...], "last_reco": str|None, "sm": SessionManager} }
sessions: Dict[str, Dict[str, Any]] = {}


# -------------------- session helpers --------------------
def _get_session(sid: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """
    Backward compatible session getter.
    Ensures every session always has:
      - history
      - last_reco
      - sm (SessionManager)
    """
    if not sid or sid not in sessions:
        sid = uuid.uuid4().hex[:12]
        sessions[sid] = {"history": [], "last_reco": None, "sm": SessionManager()}
    else:
        if "history" not in sessions[sid]:
            sessions[sid]["history"] = []
        if "last_reco" not in sessions[sid]:
            sessions[sid]["last_reco"] = None
        if "sm" not in sessions[sid]:
            sessions[sid]["sm"] = SessionManager()
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


# -------------------- static movie pool --------------------
MOVIE_RECS = [
    ('The Shawshank Redemption', "an inspiring drama about hope and friendship"),
    ('Inception', "a mind-bending sci-fi thriller with stunning visuals"),
    ('La La Land', "a warm musical about love, dreams, and second chances"),
    ('Spider-Man: Into the Spider-Verse', "a fun, stylish animated adventure"),
    ('Knives Out', "a clever, modern whodunit with sharp humor"),
]


# -------------------- FAST SUMMARY (NO LLM) --------------------
_END_PHRASES = {"end session", "end", "finish session", "stop session", "quit session"}

def _one_line(s: str, max_len: int = 140) -> str:
    """Collapse whitespace and hard-trim."""
    s = (s or "").strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip() + "..."
    return s

def _safe_title_from_transcript(transcript: List[Dict[str, Any]]) -> str:
    """Pick a human-looking title from the first meaningful user message."""
    for t in transcript:
        if (t.get("role") or "").lower() == "user":
            ut = (t.get("text") or "").strip()
            if not ut:
                continue
            if _normalize(ut) in _END_PHRASES:
                continue
            # Prefer tool-like intents as the title (e.g., "search arxiv transformers")
            return _one_line(ut, 80)
    return "Session Summary"

def _infer_focus_line(transcript: List[Dict[str, Any]], hits: List[Dict[str, Any]]) -> str:
    """
    Create a natural 'focus' sentence from the user's goal + retrieval context.
    Deterministic, no model calls.
    """
    # Try to infer from last meaningful user query
    last_user = ""
    for t in reversed(transcript):
        if (t.get("role") or "").lower() == "user":
            ut = (t.get("text") or "").strip()
            if ut and _normalize(ut) not in _END_PHRASES:
                last_user = ut
                break

    # Heuristic: arxiv tool
    if last_user.lower().startswith("search arxiv "):
        q = last_user[len("search arxiv "):].strip()
        if q:
            return f"This session focused on finding relevant arXiv papers for: {q}."

    # If we have hits, mention evidence-backed browsing
    if hits:
        return "This session focused on gathering evidence-backed references from retrieved paper fragments."

    if last_user:
        return f"This session focused on: {_one_line(last_user, 90)}."
    return "This session captured the key discussion points and supporting evidence."

def _evidence_lines(hits: List[Dict[str, Any]], k: int = 2) -> List[str]:
    """
    Turn retrieval hits into readable evidence bullets.
    Avoid dumping huge fragments; keep it note-like.
    """
    out: List[str] = []
    for h in hits[:k]:
        doc_id = _one_line(h.get("doc_id", "unknown"), 90)
        chunk_id = h.get("chunk_id", "unknown")
        content = _one_line(h.get("content", ""), 170)

        # Light cleanup so evidence looks like a note, not raw OCR
        content = content.replace("ﬁ", "fi").replace("ﬀ", "ff").replace("∼", "~")
        out.append(f"- {doc_id} ({chunk_id}): {content}")
    if not out:
        out = ["- (No retrieval evidence was recorded in this session.)"]
    return out

def _summary_bullets(transcript: List[Dict[str, Any]], hits: List[Dict[str, Any]]) -> List[str]:
    """
    Natural bullets: goal -> what was found -> suggested next step.
    """
    # Counts (kept, but phrased naturally)
    num_turns = len(transcript)
    num_hits = len(hits)

    bullets: List[str] = []
    bullets.append(_infer_focus_line(transcript, hits))

    if num_hits > 0:
        bullets.append(f"Retrieved {num_hits} paper fragment(s) to support follow-up reading and comparison.")
    else:
        bullets.append("No external evidence snippets were recorded; the notes reflect only the dialogue context.")

    # Next step guidance
    if num_hits >= 2:
        bullets.append("Next step: open 1–2 top-ranked papers and write a deeper summary (methods, results, takeaways).")
    else:
        bullets.append("Next step: refine the query and retrieve more targeted evidence before writing a deeper summary.")

    # Ensure bullets are short and clean
    return [_one_line(b, 160) for b in bullets]

def _questions(transcript: List[Dict[str, Any]], hits: List[Dict[str, Any]]) -> List[str]:
    """
    Deterministic questions that feel less templated.
    """
    # If user did arxiv search, ask about selection criteria
    last_user = ""
    for t in reversed(transcript):
        if (t.get("role") or "").lower() == "user":
            ut = (t.get("text") or "").strip()
            if ut and _normalize(ut) not in _END_PHRASES:
                last_user = ut
                break

    qs: List[str] = []
    if last_user.lower().startswith("search arxiv "):
        qs.append("Which of the retrieved papers best matches what you want to achieve?")
        qs.append("Do you want a high-level overview or a technical deep dive (methods + results)?")
    else:
        qs.append("What is the main decision or output you want from this session?")
        qs.append("What extra evidence would make the final write-up stronger (more papers, data, or examples)?")

    return [_one_line(q, 120) for q in qs]

def _fast_summary_from_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Produce a demo-friendly session note instantly (no model call).
    Returns exactly: {"title": ..., "summary_text": ...}
    """
    transcript = payload.get("transcript", []) or []
    hits = payload.get("retrieval_hits", []) or []

    title = _safe_title_from_transcript(transcript)
    bullets = _summary_bullets(transcript, hits)
    qs = _questions(transcript, hits)
    evidence = _evidence_lines(hits, k=2)

    summary_text = (
        "SUMMARY:\n"
        + "\n".join([f"- {b}" for b in bullets])
        + "\n\nQUESTIONS:\n"
        + "\n".join([f"- {q}" for q in qs])
        + "\n\nKEY EVIDENCE:\n"
        + "\n".join(evidence)
    )

    return {"title": title, "summary_text": summary_text}



# -------------------- startup --------------------
@app.on_event("startup")
async def _warmup():
    # Start fast: do not load heavy models at startup.
    # Models will be loaded lazily on first use.
    return



# -------------------- routes --------------------
@app.get("/ping")
def ping():
    return {"ok": True}

@app.post("/asr")
async def asr(file: UploadFile = File(...)):
    global _asr
    if _asr is None:
        _asr = load_asr()

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio upload")
        
        try:
            text = transcribe_bytes(_asr, audio_bytes)
        except TypeError:
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
    sm: SessionManager = sess["sm"]

    user_text = (body.text or "").strip()
    if not user_text:
        return {"text": "", "session_id": session_id}

    # Legacy: keep short rolling history (used for small-talk prompt)
    _push_history(sess, "user", user_text)

    # NEW: record user turn in session manager
    sm.add_user_turn(user_text)

    # NEW: end session trigger (FAST summary)
    if sm.should_end_session(user_text):
        payload = sm.build_summary_payload()

        summary = _fast_summary_from_payload(payload)
        sm.set_final_summary(summary["title"], summary["summary_text"])

        result = {
            "text": f'Ended session. Generated summary: "{summary["title"]}".',
            "session_id": session_id,
            "summary": summary,
            "summary_payload_stats": payload.get("stats", {}),
        }

        # Reset session state for next run
        sm.reset()
        sess["history"] = []
        sess["last_reco"] = None

        return result

    # --- intent flags / routing switches ---
    is_time = _is_time_question(user_text)
    is_hello = _is_greeting(user_text)
    is_movie = _is_movie_intent(user_text)

    # ---------- 1) Simple rule tools ----------
    if is_time:
        now = datetime.now().strftime("%H:%M")
        assistant_text = f"The current time is {now}."
        sm.add_assistant_turn(assistant_text)
        _push_history(sess, "assistant", assistant_text)
        return {"text": assistant_text, "session_id": session_id}

    if is_hello:
        assistant_text = "Hi! I'm good — how can I help?"
        sm.add_assistant_turn(assistant_text)
        _push_history(sess, "assistant", assistant_text)
        return {"text": assistant_text, "session_id": session_id}

    if _refers_previous_movie(user_text) and sess.get("last_reco"):
        title = sess["last_reco"]
        assistant_text = (
            f'I suggested "{title}" because it is widely praised for its storytelling and emotional impact. '
            "It is an easy, high-quality pick for most moods."
        )
        sm.add_assistant_turn(assistant_text)
        _push_history(sess, "assistant", assistant_text)
        return {"text": assistant_text, "session_id": session_id}

    if re.search(r"\bfavou?rite\b|\bwhat.?do.?you.?like\b", user_text, re.IGNORECASE):
        short_ans = (
            "I do not have personal tastes, but sushi and pizza are among the most popular foods worldwide. "
            "What about you?"
        )
        sm.add_assistant_turn(short_ans)
        _push_history(sess, "assistant", short_ans)
        return {"text": short_ans, "session_id": session_id}

    if is_movie:
        title, blurb = random.choice(MOVIE_RECS)
        assistant_text = f'Try "{title}" — {blurb}.'
        sess["last_reco"] = title
        sm.add_assistant_turn(assistant_text)
        _push_history(sess, "assistant", assistant_text)
        return {"text": assistant_text, "session_id": session_id}

    # ---------- 2) Tool routing ----------
    lower = user_text.lower().strip()

    # Tool 1: calculate(expression)
    if lower.startswith("calculate "):
        expr = user_text[len("calculate "):].strip()
        try:
            data = calculate(expr)
            if "result" in data:
                assistant_text = f"The answer is {data['result']}."
            else:
                assistant_text = f"I could not compute that: {data.get('error', 'unknown error')}."
        except Exception as e:
            assistant_text = f"Calculator failed: {e}"

        sm.add_assistant_turn(assistant_text)
        _push_history(sess, "assistant", assistant_text)
        return {"text": assistant_text, "session_id": session_id}

    # Tool 2: search_arxiv(query)
    if lower.startswith("search arxiv "):
        query = user_text[len("search arxiv "):].strip()
        try:
            data = search_arxiv(query)
            if "error" in data:
                assistant_text = f"Search failed: {data['error']}"
                sm.add_assistant_turn(assistant_text)
                _push_history(sess, "assistant", assistant_text)
                return {"text": assistant_text, "session_id": session_id}

            matches = data.get("matches", [])
            if not matches:
                assistant_text = "No matching arxiv chunks found."
                sm.add_assistant_turn(assistant_text)
                _push_history(sess, "assistant", assistant_text)
                return {"text": assistant_text, "session_id": session_id}

            # NEW: log retrieval hits into session manager
            hits: List[Dict[str, Any]] = []
            for idx, c in enumerate(matches[:6], start=1):
                title = c.get("title", "(no title)")
                text_snip = (c.get("text", "") or "").strip()
                if len(text_snip) > 1200:
                    text_snip = text_snip[:1200].rstrip() + "..."
                hits.append({
                    "source": "arxiv",
                    "doc_id": title,
                    "chunk_id": f"match_{idx}",
                    "content": text_snip,
                    "score": c.get("score", None),
                    "meta": {k: v for k, v in c.items() if k not in {"title", "text", "score"}},
                })

            sm.add_retrieval(query=query, hits=hits)

            # User-facing response
            lines = []
            for h in hits[:3]:
                t = h["doc_id"]
                sn = h["content"]
                if len(sn) > 90:
                    sn = sn[:90].rstrip() + "..."
                lines.append(f"- {t}: {sn}")
            assistant_text = "Here are some arxiv matches:\n" + "\n".join(lines)

        except Exception as e:
            assistant_text = f"Arxiv search failed: {e}"

        sm.add_assistant_turn(assistant_text)
        _push_history(sess, "assistant", assistant_text)
        return {"text": assistant_text, "session_id": session_id}

    # ---------- 3) DEFAULT: LLM SMALL-TALK / GENERAL QA ----------
    history_block = _history_to_prompt(sess, max_turns=3)
    system_rules = (
        "You are a concise, friendly assistant.\n"
        "Rules: Do not mention being an AI or language model; speak naturally like a person.\n"
        "Respond in one or two short sentences; no emojis; no hashtags; "
        "do NOT write 'User:' lines; do NOT continue any dialogue template.\n\n"
    )
    prompt = (
        f"{system_rules}"
        f"{history_block}\n"
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

    sm.add_assistant_turn(ans)
    _push_history(sess, "assistant", ans)
    return {"text": ans, "session_id": session_id}


@app.post("/tts")
async def tts(body: TTSBody) -> Response:
    text = (body.text or "").strip()

    # Fast guard: skip too short/too long text to avoid slow/failing TTS
    if not text or len(text) < 3 or len(text) > 200:
        return Response(status_code=204)

    voice = body.voice or DEFAULT_VOICE
    rate = body.rate or DEFAULT_RATE
    pitch = body.pitch or DEFAULT_PITCH

    try:
        mp3_bytes = await synthesize_mp3_async(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
        )

        if not mp3_bytes:
            return Response(status_code=204)

        headers = {
            "Content-Disposition": 'inline; filename="tts.mp3"',
            "Cache-Control": "no-store",
        }
        return Response(content=mp3_bytes, media_type="audio/mpeg", headers=headers)

    except Exception:
        return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
