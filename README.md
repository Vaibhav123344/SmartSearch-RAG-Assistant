# 🔍 SmartSearch RAG Assistant

A powerful Streamlit-based chatbot that combines **document-based retrieval (RAG)** with **real-time web search** using Groq’s ultra-fast LLMs.

## 🚀 Features

- ✅ **Groq + Langchain**: Uses `Gemma2-9b-It` model via Groq API.
- 📄 **PDF/TXT Ingestion**: Upload files or load from a URL.
- 🧠 **Vector Store with Contextual Memory**: Built with Chroma and HuggingFace MiniLM embeddings.
- 🌐 **Web Search Support**: Falls back to DuckDuckGo or Wikipedia if needed.
- 💬 **Conversational Memory**: Maintains the last 6 turns of dialogue.
- 🔊 **Voice I/O**: Includes both speech-to-text (Whisper) and text-to-speech (gTTS).
- 📊 **Analytics Dashboard**: Tracks question count, latency, and source used.

## 📦 Tech Stack

| Layer        | Tool/Service                    |
|--------------|---------------------------------|
| UI           | Streamlit                       |
| LLM          | Groq (Gemma2-9b-It)             |
| Embeddings   | HuggingFace (MiniLM-L6-v2)      |
| Vector DB    | Chroma                          |
| Web Search   | DuckDuckGo + Wikipedia          |
| Voice        | gTTS + HuggingFace Whisper      |
| Data Format  | LangChain Documents & Schema    |

## 🛠 Installation

1. Clone the repo:

```bash
git clone https://github.com/Vaibhav123344/SmartSearch-RAG-Assistant.git
cd SmartSearch-RAG-Assistant
