import os
import time
from datetime import datetime
import tempfile

import streamlit as st

from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import WikipediaAPIWrapper

from pydantic import BaseModel, Field
from typing import Literal

from transformers import pipeline
from gtts import gTTS

from dotenv import load_dotenv
import pandas as pd

load_dotenv()

ENABLE_AUTH = False  # Change to True to enable token authentication
if ENABLE_AUTH:
    token = st.sidebar.text_input("Enter access token", type="password")
    if not token or token != "secret-token":
        st.warning("Invalid or missing token.")
        st.stop()


st.sidebar.header("Configuration")
groq_api_key = os.getenv("GROQ_API_KEY") or st.sidebar.text_input("Groq API Key", type="password")
if not groq_api_key:
    st.sidebar.warning("Please set your GROQ_API_KEY environment variable or enter it here.")
    st.stop()


llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "embeddings" not in st.session_state:
    # Use all-MiniLM-L6-v2 for embeddings
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "stats" not in st.session_state:
    st.session_state.stats = {"questions": 0, "queries": []}
if "sources" not in st.session_state:
    st.session_state.sources = []


class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ..., description="Choose 'vectorstore' or 'wiki_search'."
    )
structured_router = llm.with_structured_output(RouteQuery)
routing_system_prompt = (
    "You are an expert at routing user questions. "
    "The vectorstore contains domain-specific documents; use it for related questions. "
    "Use Wikipedia search for general knowledge. "
    "Return only JSON with key 'datasource'."
)
from langchain.prompts.chat import ChatPromptTemplate
route_prompt = ChatPromptTemplate.from_messages([
    ("system", routing_system_prompt),
    ("human", "{question}")
])
router_chain = route_prompt | structured_router


st.sidebar.header("Knowledge Base")
url_input = st.sidebar.text_input("Enter a URL to load")
uploaded_files = st.sidebar.file_uploader("Upload PDF or TXT", accept_multiple_files=True)
if st.sidebar.button("Add Source"):
    added = False
    # Handle URL loading
    if url_input:
        try:
            docs = WebBaseLoader(url_input).load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=st.session_state.embeddings,
                    persist_directory="chroma_db"
                )
            else:
                st.session_state.vectorstore.add_documents(chunks)
            st.session_state.sources.append(url_input)
            st.sidebar.success(f"Loaded URL: {url_input}")
            added = True
        except Exception as e:
            st.sidebar.error(f"Error loading URL: {e}")
    # Handle file uploads
    for file in uploaded_files:
        try:
            suffix = os.path.splitext(file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.read()); tmp.flush()
                loader = PyPDFLoader(tmp.name) if suffix == ".pdf" else TextLoader(tmp.name)
                docs = loader.load()
            os.unlink(tmp.name)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=st.session_state.embeddings,
                    persist_directory="chroma_db"
                )
            else:
                st.session_state.vectorstore.add_documents(chunks)
            st.session_state.sources.append(file.name)
            st.sidebar.success(f"Loaded file: {file.name}")
            added = True
        except Exception as e:
            st.sidebar.error(f"Error loading {file.name}: {e}")
    if not added:
        st.sidebar.info("Enter a URL or upload at least one file.")

if st.session_state.sources:
    st.sidebar.markdown("**Loaded Sources:**")
    for src in st.session_state.sources:
        st.sidebar.write(f"- {src}")


st.title("🎯 RAG Chatbot")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input and processing
if user_q := st.chat_input("Enter your question..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_q)
    st.session_state.messages.append({"role": "user", "content": user_q})

    # Route question
    try:
        route = router_chain.invoke({"question": user_q})["datasource"]
    except:
        route = "vectorstore"

    # Retrieve context
    context = ""
    if route == "wiki_search":
        wiki = WikipediaAPIWrapper(top_k_results=2)
        context = wiki.run(user_q)
    else:
        if st.session_state.vectorstore:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(user_q)
            context = "\n\n".join([d.page_content for d in docs])
        else:
            docs = []

    # Build prompt
    if context:
        system_msg = (
            "You are a helpful assistant. Use the provided context to answer the question. "
            "If the context doesn't cover it, you may still use your general knowledge."
        )
        human_msg = f"Context:\n{context}\n\nQuestion: {user_q}\nAnswer:"
    else:
        system_msg = "You are a helpful assistant. Answer the question to the best of your ability."
        human_msg = user_q

    # Invoke LLM
    start = time.time()
    try:
        resp = llm.invoke([("system", system_msg), ("human", human_msg)])
        answer = resp.content
    except Exception as e:
        answer = f"LLM Error: {e}"
    latency = round(time.time() - start, 2)

    # assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Update analytics
    st.session_state.stats["questions"] += 1
    st.session_state.stats["queries"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latency": latency,
        "source": route
    })

    if st.button("🔊 Speak Answer"):
        try:
            tts = gTTS(answer)
            tts.save("answer.mp3")
            st.audio("answer.mp3", format="audio/mp3")
        except Exception as e:
            st.error(f"TTS Error: {e}")

    if st.button("🎤 Record Question"):
        audio_data = st.audio_input("Record your question", key="audio_input")
        if audio_data:
            try:
                asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
                transcript = asr(audio_data.read())["text"]
                st.write(f"Transcribed: {transcript}")
            except Exception as e:
                st.error(f"ASR Error: {e}")


st.sidebar.header("Analytics")
st.sidebar.metric("Total Questions", st.session_state.stats["questions"])
if st.session_state.stats["queries"]:
    df = pd.DataFrame(st.session_state.stats["queries"])
    st.sidebar.dataframe(df)
