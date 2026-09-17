# RAG Support Chatbot

A small RAG (Retrieval-Augmented Generation) chatbot built with LangChain + Google
Gemini, from the Practical Generative AI Workshop (Section 8 self-exercise).
Ships with both a CLI and a browser-based UI.

## How it works

```
User -> Chat Interface -> Conversation History -> Retriever -> Relevant Context -> Gemini -> Response
```

On every turn, the app retrieves the chunks of `context.txt` most relevant to the new
message, combines them with the running conversation history, and sends both to
Gemini. If the answer isn't in the document or conversation, the model says so instead
of guessing.

## Project structure

```
chatbot.py      # core RAG + memory logic (embeddings, vector store, rag_chat())
server.py       # FastAPI bridge exposing rag_chat() to the browser UI
static/         # HTML/CSS/JS frontend (index.html, style.css, script.js)
context.txt     # the source document the chatbot retrieves answers from
pyproject.toml  # dependencies (managed with uv)
```

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) installed
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Setup

```bash
uv sync
```

This creates `.venv/` and installs everything listed in `pyproject.toml`.

## Running it

### Option A — command-line chatbot

```bash
uv run python chatbot.py
```

You'll be prompted once for your Gemini API key (input is hidden, never saved to
disk). Chat in the terminal; type `quit` to exit.

### Option B — browser UI

```bash
uv run uvicorn server:app --reload
```

Then open **http://127.0.0.1:8000/**. The first request triggers the same API-key
prompt in the terminal running the server.

Use the **Reset** button in the UI to clear conversation memory without restarting the
server.

## Setting the API key without a prompt

To skip the interactive prompt (e.g. for repeated runs), export the key first:

```bash
export GOOGLE_API_KEY="your-key-here"
uv run uvicorn server:app --reload
```

## Customizing the knowledge base

Edit `context.txt` with your own document — the chatbot re-chunks, re-embeds, and
retrieves from whatever is in that file at startup. No code changes needed.
