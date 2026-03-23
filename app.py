import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pathlib import Path
import os

# Load API key
load_dotenv(dotenv_path=Path(".env"), override=True)

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="TechMart AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# ---- CACHE HEAVY COMPONENTS ----
@st.cache_resource
def load_rag_pipeline():
    # Load document
    loader = TextLoader(
        "data/techmart_policy.txt",
        encoding="utf-8"
    )
    documents = loader.load()

    # Chunk document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # Create LLM
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

    return retriever, llm

# ---- UI HEADER ----
st.title("🤖 TechMart AI Assistant")
st.caption("Ask me anything about TechMart policies!")
st.divider()

# ---- LOAD PIPELINE ----
with st.spinner("Loading TechMart knowledge base..."):
    retriever, llm = load_rag_pipeline()

# ---- SESSION STATE FOR MEMORY ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- DISPLAY CHAT HISTORY ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---- CHAT INPUT ----
user_input = st.chat_input("Ask about TechMart policies...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Retrieve relevant chunks
    relevant_docs = retriever.invoke(user_input)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Build messages
    messages = [
        ("system", f"""You are TechMart's helpful customer support assistant.
Use ONLY the context below to answer questions.
If answer not in context, say 'I don't have that information.'

CONTEXT:
{context}"""),
    ]

    # Add conversation history
    for msg in st.session_state.chat_history:
        messages.append(msg)

    # Add current question
    messages.append(("human", user_input))

    # Generate response
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | StrOutputParser()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({"context": context})
        st.write(response)

    # Save to histories
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
    st.session_state.chat_history.append(("human", user_input))
    st.session_state.chat_history.append(("assistant", response))