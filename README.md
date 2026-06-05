# Richard Feynman Digital Twin — Premium AI Agent

An advanced, highly interactive, and visually stunning digital twin of the Nobel Laureate physicist, Richard P. Feynman. This project emulates Feynman's scientific reasoning, signature speech patterns (the "Feynman Technique"), boyish personality, and physics knowledge through a state-of-the-art RAG pipeline, stateful dual-tier memory, authentic F5-TTS zero-shot voice cloning, and an ultra-premium glassmorphic web dashboard containing a live-drawing chalk blackboard and curiosity meter.

 Premium Visual Innovations
1.  The Chalk Blackboard (Signature Feature)
Every time a physics or science question is asked, Feynman "draws" the concept live on a dark green blackboard panel on the right side of the screen.

Chalk coordinate drawings: The model generates clean, ultra-compact SVG sketches following a hand-drawn chalk style (using white #F5F5EE, yellow #FFE44D, cyan #7FFFD4, and pink #FFB3C6 chalk colors).
Live Drawing Animation: Browser SVG paths are animated line-by-line using a custom @keyframes chalk-draw CSS stroke-dashoffset transition, physically drawing the diagram in real-time as if sketched live!
2.  F5-TTS Voice Cloning (Authentic Speech)
We replaced generic text-to-speech with a realistic zero-shot voice clone of Feynman.

Voice Profile: Grounded on an authentic mono reference clip feynman_ref_mono_16k.wav isolated from his lectures.
Zero-Shot Remote Inference: Integrates with a remote Gradio 5 API (mrfakename/E2-F5-TTS) via a fast, non-blocking Server-Sent Events (SSE) polling pipeline.
Session Caching: Caches uploaded reference paths on Hugging Face to reduce voice synthesis latency to just 6-8 seconds.
Graceful Local Fallback: Automatically falls back to standard Google TTS (gTTS) if the remote service is unavailable, ensuring audio outputs are resilient.
3.  Curiosity Depth Meter & Challenger
Curiosity Score (1-10): A dynamic evaluator rating the complexity of the user's inquiry, paired with a short critique comment in Feynman's voice (e.g. "Good instinct — you're asking about the right thing").
Thought Experiment Challenger: Every 3rd physics question, Feynman challenges the user back with an interactive, counterintuitive thought experiment card to stimulate deep reasoning.
 API Rate-Limit & Performance Optimization
To protect the rolling 5 RPM (Requests Per Minute) free-tier quota of the Gemini API:

Metadata Bundling (Single-Request Architecture): We consolidated the Chat Response, Curiosity Score, Feynman's Comment, and Blackboard SVG drawing into a single main chatbot API call.
XML Parsing Engine: Feynman's prompt instructs the model to append metadata tags (<curiosity_score>, <curiosity_comment>, <blackboard_svg>) at the end of its response. The app parses these tags via regex, separates the visual components for rendering, and removes them from the text shown in the bubble and sent to the TTS engine.
Throttled Memory Consolidation: Long-term memory extraction (memory.py) is throttled to run only every 3rd turn, reducing overall LLM calls.

Quick Start & Reproducibility Guide

1. Clone & Set Up Directory
2.  Configure Virtual Environment
3.  Set Up Credentials & API Keys
4.  Populate Knowledge Base (RAG)
5.  Launch the Digital Twin

System Architecture


       [ Speak Input (Mic WebM/OGG) ]      [ Text Input (Type) ]
                     │                              │
                     ▼                              ▼
             STT Transcription              st.chat_input
                     │                              │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                           Agent Core (main.py)
                                    │
                                    ▼
                          Gemini 3.5 Flash API  <── [ RAG / Memory Context ]
                                    │
                                    ├───► [ Chat Reply Text ] ──► F5-TTS ──► [ Audio Autoplay ]
                                    ├───► [ Curiosity Score / Comment ] ──► [ Gauge Widget ]
                                    └───► [ Blackboard SVG Drawing ] ──► [ Chalk CSS Animation ]
