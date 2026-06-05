"""
Memory System for Feynman Digital Twin
- Short-term: in-memory conversation history (list of messages)
- Long-term: JSON file persisted to disk across sessions
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import google.generativeai as genai

MEMORY_DIR = Path("memory_store")
MEMORY_FILE = MEMORY_DIR / "long_term_memory.json"


def init_memory():
    """Create memory directory and file if they don't exist."""
    MEMORY_DIR.mkdir(exist_ok=True)
    if not MEMORY_FILE.exists():
        default = {
            "user_facts": [],       # Facts the user has shared
            "topics_discussed": [], # Topics covered across sessions
            "user_name": None,      # User's name if shared
            "session_count": 0,     # How many sessions
            "last_session": None,   # ISO timestamp
            "notable_exchanges": [], # Key moments worth remembering
            "active_session_id": "default", # Currently active session key
            "sessions": {
                "default": {
                    "title": "Current Exploration",
                    "created_at": datetime.now().isoformat(),
                    "history": []
                }
            }
        }
        with open(MEMORY_FILE, "w") as f:
            json.dump(default, f, indent=2)


def load_long_term_memory() -> dict:
    """Load persistent memory from disk."""
    init_memory()
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def save_long_term_memory(memory: dict):
    """Save updated memory to disk."""
    init_memory()
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def update_session_metadata():
    """Update session count and timestamp on each new session."""
    memory = load_long_term_memory()
    memory["session_count"] = memory.get("session_count", 0) + 1
    memory["last_session"] = datetime.now().isoformat()
    save_long_term_memory(memory)
    return memory


def create_new_session(title: str = "New Conversation") -> str:
    """Create a new chat session and make it active."""
    memory = load_long_term_memory()
    session_id = "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    if "sessions" not in memory:
        memory["sessions"] = {}
    memory["sessions"][session_id] = {
        "title": title,
        "created_at": datetime.now().isoformat(),
        "history": []
    }
    memory["active_session_id"] = session_id
    save_long_term_memory(memory)
    return session_id


def switch_session(session_id: str):
    """Switch active chat session."""
    memory = load_long_term_memory()
    if "sessions" in memory and session_id in memory["sessions"]:
        memory["active_session_id"] = session_id
        save_long_term_memory(memory)


def delete_session(session_id: str):
    """Delete a session from disk."""
    memory = load_long_term_memory()
    if "sessions" in memory and session_id in memory["sessions"]:
        del memory["sessions"][session_id]
        if memory.get("active_session_id") == session_id:
            # Switch back to remaining or default
            remaining = list(memory["sessions"].keys())
            if remaining:
                memory["active_session_id"] = remaining[0]
            else:
                memory["active_session_id"] = "default"
                if "sessions" not in memory:
                    memory["sessions"] = {}
                memory["sessions"]["default"] = {
                    "title": "Current Exploration",
                    "created_at": datetime.now().isoformat(),
                    "history": []
                }
        save_long_term_memory(memory)


def add_user_fact(fact: str):
    """Store a fact the user revealed about themselves."""
    memory = load_long_term_memory()
    if fact not in memory["user_facts"]:
        memory["user_facts"].append(fact)
        save_long_term_memory(memory)


def add_topic_discussed(topic: str):
    """Track a topic that was discussed."""
    memory = load_long_term_memory()
    if topic not in memory["topics_discussed"]:
        memory["topics_discussed"].append(topic)
        save_long_term_memory(memory)


def set_user_name(name: str):
    """Remember the user's name."""
    memory = load_long_term_memory()
    memory["user_name"] = name
    save_long_term_memory(memory)


def add_notable_exchange(user_msg: str, summary: str):
    """Save a notable question/exchange for future reference."""
    memory = load_long_term_memory()
    exchange = {
        "date": datetime.now().isoformat()[:10],
        "user_asked": user_msg[:120],
        "summary": summary
    }
    memory["notable_exchanges"].append(exchange)
    # Keep only the last 20 notable exchanges
    memory["notable_exchanges"] = memory["notable_exchanges"][-20:]
    save_long_term_memory(memory)


def format_memory_context() -> str:
    """Format long-term memory into a string for the system prompt."""
    memory = load_long_term_memory()
    parts = []

    if memory.get("user_name"):
        parts.append(f"User's name: {memory['user_name']}")

    if memory.get("session_count", 0) > 1:
        parts.append(f"This is session #{memory['session_count']} with this user.")
        parts.append(f"Last talked: {memory.get('last_session', '')[:10]}")

    if memory.get("user_facts"):
        facts = "; ".join(memory["user_facts"][-5:])  # last 5 facts
        parts.append(f"Things I know about this user: {facts}")

    if memory.get("topics_discussed"):
        topics = ", ".join(memory["topics_discussed"][-8:])
        parts.append(f"Topics we've covered before: {topics}")

    if memory.get("notable_exchanges"):
        last = memory["notable_exchanges"][-3:]
        exchanges = " | ".join([f"[{e['date']}] {e['summary']}" for e in last])
        parts.append(f"Recent memorable exchanges: {exchanges}")

    return "\n".join(parts) if parts else ""


# ─── Short-term (conversation) memory ───────────────────────────────────────

class ConversationMemory:
    """
    Manages the in-session message history.
    Automatically trims to avoid exceeding context window.
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns  # max message pairs to keep
        self.history = []
        self.load_from_active_session()

    def load_from_active_session(self):
        """Load the active session's conversation history from disk."""
        try:
            memory = load_long_term_memory()
            active_id = memory.get("active_session_id", "default")
            sessions = memory.get("sessions", {})
            self.history = sessions.get(active_id, {}).get("history", [])
        except Exception:
            self.history = []

    def add_user(self, text: str):
        self.history.append({"role": "user", "parts": [text]})
        self._trim()
        self._save_to_disk()

    def add_model(self, text: str):
        self.history.append({"role": "model", "parts": [text]})
        self._trim()
        self._save_to_disk()

    def _trim(self):
        """Keep only the last max_turns*2 messages (pairs)."""
        max_msgs = self.max_turns * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def get_history(self) -> list:
        """Return history in Gemini-compatible format."""
        return self.history

    def clear(self):
        self.history = []
        self._save_to_disk()

    def _save_to_disk(self):
        try:
            memory = load_long_term_memory()
            active_id = memory.get("active_session_id", "default")
            if "sessions" not in memory:
                memory["sessions"] = {}
            if active_id not in memory["sessions"]:
                memory["sessions"][active_id] = {
                    "title": "New Conversation",
                    "created_at": datetime.now().isoformat(),
                    "history": []
                }
            memory["sessions"][active_id]["history"] = self.history
            
            # If the session is new and just got its first user query, generate a smart title!
            if len(self.history) == 1 and self.history[0]["role"] == "user":
                query_text = self.history[0]["parts"][0]
                title = query_text[:30] + ("..." if len(query_text) > 30 else "")
                memory["sessions"][active_id]["title"] = title
                
            save_long_term_memory(memory)
        except Exception as e:
            print(f"[Memory Engine] Failed to save conversation session to disk: {e}")

    def summary_for_long_term(self) -> str:
        """Create a brief summary of this session for long-term storage."""
        if not self.history:
            return ""
        topics = set()
        for msg in self.history:
            if msg["role"] == "user":
                text = msg["parts"][0].lower()
                for keyword in ["quantum", "physics", "feynman", "qed", "electron",
                                "light", "math", "energy", "atom", "particle",
                                "relativity", "spin", "lecture", "diagram"]:
                    if keyword in text:
                        topics.add(keyword)
        return f"Discussed: {', '.join(list(topics)[:5])}" if topics else "General conversation"


def call_gemini_with_retry(fn, *args, max_retries=5, initial_delay=2.0, backoff=2.0, **kwargs):
    import time
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            is_429 = any(term in err_str for term in ["429", "quota", "resourceexhausted", "resource exhausted", "limit reached"])
            if is_429 and attempt < max_retries - 1:
                print(f"[Retry Engine] Gemini rate limit hit. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= backoff
            else:
                raise e


def consolidate_memory_with_llm(user_msg: str, model_msg: str):
    """
    Use Gemini 2.5 Flash to analyze the single turn exchange and extract:
    - User's name (if shared)
    - New facts about the user
    - Topics discussed
    - Summary of the exchange (if it's a notable exchange)
    Then, update the long-term memory store on disk.
    """
    try:
        memory = load_long_term_memory()
        
        current_name = memory.get("user_name")
        current_facts = memory.get("user_facts", [])
        current_topics = memory.get("topics_discussed", [])
        
        prompt = f"""
You are an expert information extraction assistant for a Digital Twin of Richard Feynman.
Analyze the following single exchange between a User and Feynman.
Your goal is to extract key personal details about the User and the conversation topics to update Feynman's long-term memory.

Current knowledge about User:
- User Name: {current_name or "Unknown"}
- Known Facts about User: {json.dumps(current_facts)}
- Known Topics Discussed: {json.dumps(current_topics)}

Conversation exchange:
[User]: {user_msg}
[Feynman]: {model_msg}

Extract the following in strict JSON format:
1. "user_name": The user's name if they explicitly introduced themselves in this turn (otherwise null).
2. "new_facts": List of NEW personal facts, background, or goals the user shared about themselves. Do NOT repeat facts already in the list above. Frame them from Feynman's perspective (e.g., "Is a student at DTU", "Is working on a project about QED"). Keep them very brief. If none, return empty list [].
3. "new_topics": List of general topics discussed (e.g., "Quantum Mechanics", "QED", "Teaching", "Superfluidity", "Lock-picking"). Max 3 topics.
4. "notable_exchange_summary": If this exchange was a deep physical discussion, a beautiful analogy, or a personal breakthrough, write a very brief 1-sentence summary of the exchange (e.g., "We talked about why spin has no classical rotating counterpart"). If it is just casual greeting, chit-chat, or follow-up, return null.

Response MUST be a valid JSON object only, with no markdown code blocks, backticks, or trailing commas.
Example:
{{
  "user_name": "Arjun",
  "new_facts": ["Is studying physics at DTU"],
  "new_topics": ["Quantum Mechanics", "Spin"],
  "notable_exchange_summary": "Discussed the physical meaning of quantum spin and why it's not a physical rotation."
}}
"""
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = call_gemini_with_retry(
            model.generate_content,
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        result = json.loads(response.text.strip())
        
        new_name = result.get("user_name")
        if new_name and not current_name:
            memory["user_name"] = new_name
            print(f"[Memory Engine] Learned user name: {new_name}")
            
        new_facts = result.get("new_facts", [])
        for fact in new_facts:
            if fact not in memory["user_facts"]:
                memory["user_facts"].append(fact)
                print(f"[Memory Engine] Added fact: {fact}")
                
        new_topics = result.get("new_topics", [])
        for topic in new_topics:
            if topic not in memory["topics_discussed"]:
                memory["topics_discussed"].append(topic)
                print(f"[Memory Engine] Added topic: {topic}")
                
        summary = result.get("notable_exchange_summary")
        if summary:
            exchange = {
                "date": datetime.now().isoformat()[:10],
                "user_asked": user_msg[:120],
                "summary": summary
            }
            if "notable_exchanges" not in memory:
                memory["notable_exchanges"] = []
            memory["notable_exchanges"].append(exchange)
            memory["notable_exchanges"] = memory["notable_exchanges"][-20:]
            print(f"[Memory Engine] Logged notable exchange: {summary}")
            
        save_long_term_memory(memory)
        
    except Exception as e:
        print(f"[Memory Engine] LLM memory consolidation failed: {e}")