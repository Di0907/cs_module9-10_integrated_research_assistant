from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SessionState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    RETRIEVING = "RETRIEVING"
    ANSWERING = "ANSWERING"
    SUMMARIZING = "SUMMARIZING"
    SYNCING = "SYNCING"


@dataclass
class Turn:
    role: str  # "user" | "assistant" | "system"
    text: str
    ts: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class RetrievalHit:
    """
    A normalized retrieval hit. Keep this format stable for summarization + Notion.
    """
    source: str                 # e.g. "local_chunks", "arxiv", "pdf"
    doc_id: str                 # e.g. paper id, filename, url, etc.
    chunk_id: str               # e.g. "c12" or "page3_chunk4"
    content: str                # the actual snippet
    score: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalEvent:
    query: str
    hits: List[RetrievalHit] = field(default_factory=list)
    ts: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SessionConfig:
    """
    Customize end phrases and caps to keep summaries clean.
    """
    end_phrases: List[str] = field(default_factory=lambda: [
        "end session",
        "finish session",
        "stop session",
        "end",
        "stop",
        "that's all",
        "that is all",
        "bye",
        "goodbye",
        "quit",
        "exit",
    ])
    max_turns_kept: int = 200
    max_hits_kept: int = 80
    max_chars_per_hit: int = 1200  # avoid over-long snippets in payload


class SessionManager:
    """
    Keeps conversation-level state so you can:
    - generate session summary after user ends the session
    - sync final output to Notion
    """

    def __init__(self, config: Optional[SessionConfig] = None):
        self.config = config or SessionConfig()
        self.state: SessionState = SessionState.IDLE
        self.session_id: str = self._new_session_id()
        self.turns: List[Turn] = []
        self.retrievals: List[RetrievalEvent] = []

        # Optional: store final artifacts
        self.final_summary_text: Optional[str] = None
        self.final_title: Optional[str] = None
        self.notion_page_url: Optional[str] = None

    # -------------------
    # Lifecycle
    # -------------------
    def start_if_needed(self) -> None:
        if self.state == SessionState.IDLE:
            self.state = SessionState.LISTENING

    def reset(self) -> None:
        self.state = SessionState.IDLE
        self.session_id = self._new_session_id()
        self.turns.clear()
        self.retrievals.clear()
        self.final_summary_text = None
        self.final_title = None
        self.notion_page_url = None

    def set_state(self, new_state: SessionState) -> None:
        self.state = new_state

    # -------------------
    # Turn logging
    # -------------------
    def add_user_turn(self, text: str) -> None:
        self.start_if_needed()
        self._append_turn(role="user", text=text)
        self.state = SessionState.RETRIEVING

    def add_assistant_turn(self, text: str) -> None:
        self._append_turn(role="assistant", text=text)
        self.state = SessionState.LISTENING

    def add_system_turn(self, text: str) -> None:
        self._append_turn(role="system", text=text)

    def _append_turn(self, role: str, text: str) -> None:
        self.turns.append(Turn(role=role, text=text))
        # cap turns
        if len(self.turns) > self.config.max_turns_kept:
            self.turns = self.turns[-self.config.max_turns_kept :]

    # -------------------
    # Retrieval logging
    # -------------------
    def add_retrieval(self, query: str, hits: List[Dict[str, Any]] | List[RetrievalHit]) -> None:
        """
        Accept either:
          - a list of RetrievalHit objects, OR
          - a list of dicts that can be normalized to RetrievalHit.

        Each hit dict recommended keys:
          source, doc_id, chunk_id, content, score(optional), meta(optional)
        """
        normalized: List[RetrievalHit] = []
        for h in hits:
            if isinstance(h, RetrievalHit):
                hit_obj = h
            else:
                hit_obj = RetrievalHit(
                    source=str(h.get("source", "unknown")),
                    doc_id=str(h.get("doc_id", "unknown")),
                    chunk_id=str(h.get("chunk_id", "unknown")),
                    content=str(h.get("content", "")),
                    score=h.get("score", None),
                    meta=dict(h.get("meta", {})) if h.get("meta", {}) is not None else {},
                )

            # trim content to avoid giant payloads
            if len(hit_obj.content) > self.config.max_chars_per_hit:
                hit_obj.content = hit_obj.content[: self.config.max_chars_per_hit] + "…"

            normalized.append(hit_obj)

        self.retrievals.append(RetrievalEvent(query=query, hits=normalized))
        # cap hits overall by trimming old retrieval events
        self._cap_retrievals()
        self.state = SessionState.ANSWERING

    def _cap_retrievals(self) -> None:
        """
        Keep retrieval events but cap total hit count across events.
        """
        total_hits = sum(len(r.hits) for r in self.retrievals)
        if total_hits <= self.config.max_hits_kept:
            return

        # Drop oldest events until within cap
        while self.retrievals and total_hits > self.config.max_hits_kept:
            dropped = self.retrievals.pop(0)
            total_hits -= len(dropped.hits)

    # -------------------
    # Session end detection
    # -------------------
    def should_end_session(self, user_text: str) -> bool:
        """
        Light heuristic: check if user text matches any end phrase.
        You can extend this later with LLM-based intent detection if needed.
        """
        if not user_text:
            return False
        t = user_text.strip().lower()
        # normalize punctuation
        t = t.replace(".", "").replace("!", "").replace("?", "").replace(",", "")
        return any(phrase in t for phrase in self.config.end_phrases)

    # -------------------
    # Summary payload (for Summary Agent)
    # -------------------
    def build_summary_payload(self) -> Dict[str, Any]:
        """
        Build a clean, structured input for the summarization step.
        This is what you send into your LLM as a JSON or structured prompt.
        """
        self.state = SessionState.SUMMARIZING

        transcript = [
            {"role": turn.role, "text": turn.text, "ts": turn.ts}
            for turn in self.turns
        ]

        # Flatten top hits (keep order by time, and within each retrieval keep list order)
        flat_hits: List[Dict[str, Any]] = []
        for r in self.retrievals:
            for h in r.hits:
                flat_hits.append({
                    "source": h.source,
                    "doc_id": h.doc_id,
                    "chunk_id": h.chunk_id,
                    "score": h.score,
                    "content": h.content,
                    "meta": h.meta,
                    "retrieval_query": r.query,
                    "retrieval_ts": r.ts,
                })

        payload = {
            "session_id": self.session_id,
            "state": self.state.value,
            "transcript": transcript,
            "retrieval_hits": flat_hits,
            "stats": {
                "turns": len(self.turns),
                "retrieval_events": len(self.retrievals),
                "retrieval_hits": len(flat_hits),
            },
        }
        return payload

    # -------------------
    # Final artifact setters (optional but handy)
    # -------------------
    def set_final_summary(self, title: str, summary_text: str) -> None:
        self.final_title = title
        self.final_summary_text = summary_text

    def set_notion_result(self, page_url: str) -> None:
        self.notion_page_url = page_url
        self.state = SessionState.SYNCING

    # -------------------
    # Debug / export
    # -------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "turns": [asdict(t) for t in self.turns],
            "retrievals": [
                {
                    "query": r.query,
                    "ts": r.ts,
                    "hits": [asdict(h) for h in r.hits],
                }
                for r in self.retrievals
            ],
            "final_title": self.final_title,
            "final_summary_text": self.final_summary_text,
            "notion_page_url": self.notion_page_url,
        }

    @staticmethod
    def _new_session_id() -> str:
        # simple readable id; you can replace with uuid4 if preferred
        return datetime.utcnow().strftime("session_%Y%m%d_%H%M%S")
