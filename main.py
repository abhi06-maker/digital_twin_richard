

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import google.generativeai as genai
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown

from persona import get_persona_prompt
from memory import ConversationMemory, format_memory_context, update_session_metadata, consolidate_memory_with_llm
from rag import rag

# ─── Setup ───────────────────────────────────────────────────────────────────

load_dotenv()
console = Console()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    console.print("[red]Error: GEMINI_API_KEY not found.[/red]")
    console.print("Create a .env file with: GEMINI_API_KEY=your_key_here")
    console.print("Get your key at: https://aistudio.google.com/app/apikey")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

# ─── Keyword-based topic extraction (simple heuristic) ───────────────────────

TOPIC_KEYWORDS = {
    "quantum mechanics": ["quantum", "wave function", "superposition", "uncertainty"],
    "QED": ["qed", "quantum electrodynamics", "photon", "electron interaction"],
    "Feynman diagrams": ["feynman diagram", "diagram", "virtual particle"],
    "path integrals": ["path integral", "sum over histories", "action"],
    "physics of everyday life": ["why is the sky", "how does fire", "rainbow", "ice"],
    "nanotechnology": ["nano", "plenty of room", "small scale"],
    "Challenger disaster": ["challenger", "shuttle", "o-ring"],
    "computing": ["computer", "quantum computer", "computation"],
    "teaching": ["how to learn", "understanding", "explain"],
}

def extract_topics(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(topic)
    return found

def extract_user_name(text: str) -> str | None:
    """Simple heuristic to detect if user shared their name."""
    import re
    patterns = [
        r"(?:my name is|i'm|i am|call me)\s+([A-Z][a-z]+)",
        r"^([A-Z][a-z]+) here",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

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

# ─── Agent class ─────────────────────────────────────────────────────────────

class FeynmanAgent:
    def __init__(self):
        self.conv_memory = ConversationMemory(max_turns=20)
        self.session_memory = update_session_metadata()
        self.rag_available = rag.is_available()

        # Add relaxed safety settings to prevent mid-sentence cutoffs
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]

        self.model = genai.GenerativeModel(
            model_name="gemini-3.5-flash",
            generation_config=genai.GenerationConfig(
                temperature=0.85,
                top_p=0.95,
            ),
            safety_settings=self.safety_settings
        )

    def _build_system_prompt(self, user_query: str) -> str:
        """Build the full system prompt for this turn."""
        # Retrieve relevant context from RAG
        retrieved = ""
        if self.rag_available:
            retrieved = rag.retrieve(user_query)

        # Get long-term memory context
        memory_ctx = format_memory_context()

        return get_persona_prompt(
            retrieved_context=retrieved,
            memory_context=memory_ctx
        )
    
    def transcribe_audio(self, audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
        """Uses Gemini 2.5 Flash to transcribe voice input."""
        try:
            response = call_gemini_with_retry(
                self.model.generate_content,
                [
                    "Transcribe this audio accurately. Output ONLY the transcribed text without any extra commentary.",
                    {"mime_type": mime_type, "data": audio_bytes}
                ]
            )
            return response.text.strip()
        except Exception as e:
            print(f"[!] Audio transcription failed: {e}")
            raise e

    def chat(self, user_input: str) -> str:
        """Send a message and get Feynman's response."""
        # Add user message to short-term memory
        self.conv_memory.add_user(user_input)

        # Build system prompt with RAG + long-term memory context
        system_prompt = self._build_system_prompt(user_input)

        # Build a dynamic model instance using system_instruction
        model = genai.GenerativeModel(
            model_name="gemini-3.5-flash",
            generation_config=genai.GenerationConfig(
                temperature=0.85,
                top_p=0.95,
            ),
            safety_settings=self.safety_settings,
            system_instruction=system_prompt
        )

        # Start a new chat session each turn (stateless, we manage history)
        chat_session = model.start_chat(history=self.conv_memory.get_history()[:-1])

        # Send current user message
        response = call_gemini_with_retry(chat_session.send_message, user_input)
        reply = response.text

        # Add to short-term memory
        self.conv_memory.add_model(reply)

        # Perform LLM-driven memory consolidation every 3rd turn to protect rate limits
        history = self.conv_memory.get_history()
        user_msg_count = len([m for m in history if m.get("role") == "user"])
        if user_msg_count % 3 == 0:
            consolidate_memory_with_llm(user_input, reply)

        return reply

    def reset_conversation(self):
        """Clear conversation history (but keep long-term memory)."""
        self.conv_memory.clear()

# ─── CLI Interface ────────────────────────────────────────────────────────────

def print_welcome(rag_available: bool, doc_count: int):
    rag_status = f"[green]✓ RAG active ({doc_count} chunks)[/green]" if rag_available else "[yellow]⚠ RAG not available (run ingest.py to add documents)[/yellow]"
    
    console.print(Panel(
        f"""[bold yellow]Richard P. Feynman[/bold yellow] — Digital Twin
[dim]Nobel Prize in Physics, 1965[/dim]

{rag_status}

[dim]Commands: 'quit' to exit, 'reset' to clear conversation history[/dim]""",
        title="[bold]Feynman Digital Twin[/bold]",
        border_style="yellow"
    ))
    console.print()

def main():
    agent = FeynmanAgent()
    doc_count = rag.doc_count()
    print_welcome(agent.rag_available, doc_count)

    while True:
        try:
            # Get user input
            console.print("[bold cyan]You:[/bold cyan] ", end="")
            user_input = input().strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "bye"):
                console.print("\n[dim]Feynman nods and returns to his bongos.[/dim]")
                break

            if user_input.lower() == "reset":
                agent.reset_conversation()
                console.print("[dim]Conversation history cleared. Long-term memory preserved.[/dim]\n")
                continue

            # Get response
            with console.status("[dim]Feynman is thinking...[/dim]", spinner="dots"):
                response = agent.chat(user_input)

            # Print response
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold yellow]Feynman[/bold yellow]",
                border_style="yellow",
                padding=(1, 2)
            ))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Session ended.[/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[dim]Try again or type 'quit' to exit.[/dim]")

if __name__ == "__main__":
    main()
