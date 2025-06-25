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
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper

from pydantic import BaseModel, Field
from typing import Literal

from transformers import pipeline
from gtts import gTTS
import pandas as pd

# ----------------------------------------
# Environment & API Keys
# ----------------------------------------
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Please set the GROQ_API_KEY environment variable.")
    st.stop()

llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")

# ----------------------------------------
# Session State Initialization
# ----------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "stats" not in st.session_state:
    st.session_state.stats = {"questions": 0, "queries": []}
if "sources" not in st.session_state:
    st.session_state.sources = []

# ----------------------------------------
# Define Router Schema
# ----------------------------------------
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search", "duckduckgo"]

router = llm.with_structured_output(RouteQuery)
routing_prompt = (
    "Route user questions to one of: vectorstore (domain docs), wiki_search, or duckduckgo (real-time web search)."  
    "Return JSON {'datasource': ...}."
)
from langchain.prompts.chat import ChatPromptTemplate
route_chain = ChatPromptTemplate.from_messages([
    ("system", routing_prompt),
    ("human", "{question}")
]) | router

# ----------------------------------------
# Sidebar: Load Knowledge Sources
# ----------------------------------------
st.sidebar.header("Knowledge Base & Sources")
url_input = st.sidebar.text_input("Enter URL to index (optional)")
uploaded_files = st.sidebar.file_uploader("Upload PDF/TXT (optional)", accept_multiple_files=True)
if st.sidebar.button("Add & Index Sources"):
    loaded = False
    # URL
    if url_input:
        try:
            docs = WebBaseLoader(url_input).load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            if not st.session_state.vectorstore:
                st.session_state.vectorstore = Chroma.from_documents(chunks, st.session_state.embeddings, persist_directory="chroma_db")
            else:
                st.session_state.vectorstore.add_documents(chunks)
            st.session_state.sources.append(url_input)
            loaded = True
        except Exception as e:
            st.sidebar.error(f"URL load error: {e}")
    # Files
    for f in uploaded_files:
        try:
            suffix = os.path.splitext(f.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read()); tmp.flush()
                loader = PyPDFLoader(tmp.name) if suffix == ".pdf" else TextLoader(tmp.name)
                docs = loader.load()
            os.unlink(tmp.name)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            if not st.session_state.vectorstore:
                st.session_state.vectorstore = Chroma.from_documents(chunks, st.session_state.embeddings, persist_directory="chroma_db")
            else:
                st.session_state.vectorstore.add_documents(chunks)
            st.session_state.sources.append(f.name)
            loaded = True
        except Exception as e:
            st.sidebar.error(f"File load error: {e}")
    if not loaded:
        st.sidebar.info("No sources indexed. Add a URL or upload files.")

if st.session_state.sources:
    st.sidebar.markdown("**Indexed Sources:**")
    for src in st.session_state.sources:
        st.sidebar.write(f"- {src}")

# ----------------------------------------
# Main Chat Interface
# ----------------------------------------
st.title("RAG Chatbot ")

# Display chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# Input
query = st.chat_input("Ask your question...")
if query:
    # append user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)
    
    # routing
    try:
        route = route_chain.invoke({"question": query})["datasource"]
    except:
        route = "vectorstore"

    # get context or web results
    context = ""
    if route == "vectorstore" and st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k":3})
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])
    elif route == "wiki_search":
        context = WikipediaAPIWrapper(top_k_results=2).run(query)
    else:
        # duckduckgo search
        ddg = DuckDuckGoSearchAPIWrapper()
        context = ddg.run(query)

    # build conversation memory + prompt
    history = "\n".join([f"User: {m['content']}" if m['role']=='user' else f"Assistant: {m['content']}" for m in st.session_state.messages[-6:]])
    if context:
        system_msg = (
            "You are a knowledgeable assistant. Use the conversation history and context below to answer. "
            "If uncertain, provide best attempt using both context and history."
        )
        user_msg = f"History:\n{history}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    else:
        system_msg = "You are a knowledgeable assistant. Use conversation history to answer the question."
        user_msg = f"History:\n{history}\n\nQuestion: {query}\nAnswer:"

    # call LLM
    start = time.time()
    resp = llm.invoke([("system", system_msg), ("human", user_msg)])
    answer = resp.content
    latency = round(time.time() - start,2)

    # display answer
    with st.chat_message("assistant"): st.markdown(answer)
    st.session_state.messages.append({"role":"assistant","content":answer})

    # analytics
    st.session_state.stats["questions"] += 1
    st.session_state.stats["queries"].append({"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"latency":latency,"route":route})

# Sidebar Analytics
st.sidebar.header("Analytics")
st.sidebar.metric("Total Queries", st.session_state.stats["questions"])
if st.session_state.stats["queries"]:
    df = pd.DataFrame(st.session_state.stats["queries"])
    st.sidebar.dataframe(df)
