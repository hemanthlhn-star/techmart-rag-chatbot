# TechMart RAG Chatbot 🤖
## 🌐 Live Demo
👉 **[Try the TechMart AI Assistant here!]
(https://techmart-rag-chatbot.streamlit.app)**
A conversational AI chatbot built using 
Retrieval Augmented Generation (RAG) that answers 
customer questions about TechMart policies accurately 
— without hallucinating.

## What I Built
An end-to-end GenAI pipeline that:
- Loads real company policy documents
- Chunks and embeds them into a vector database
- Retrieves relevant context for every user question
- Generates accurate answers using a Large Language Model
- Remembers full conversation history

## Why RAG?
Traditional LLMs don't know about your company's 
internal data. RAG solves this by retrieving relevant 
documents before generating answers — eliminating 
hallucinations and giving precise, trustworthy responses.

## Architecture
```
User Question
      ↓
Convert to Embedding
      ↓
Search ChromaDB Vector Database
      ↓
Retrieve Top 3 Relevant Chunks
      ↓
Build Prompt with Context + History
      ↓
Groq LLaMA 3 generates Answer
      ↓
Save to Conversation Memory
```

## Tech Stack
| Component | Technology |
|---|---|
| Language | Python 3.14 |
| LLM | LLaMA 3.3 70B via Groq API |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| Memory | Conversation history list |
| Environment | Jupyter Notebook in VS Code |

## Project Structure
```
techmart-rag-chatbot/
├── data/
│   └── techmart_policy.txt    ← Company policy documents
├── src/
│   └── chatbot.ipynb          ← Main chatbot notebook
├── .env                       ← API keys (not uploaded)
├── .gitignore                 ← Protects sensitive files
├── requirements.txt           ← Project dependencies
└── README.md                  ← You are here!
```

## Setup Instructions

### 1. Clone the repository
git clone https://github.com/hemanthlhn-star/techmart-rag-chatbot.git
cd techmart-rag-chatbot

### 2. Install dependencies
pip install -r requirements.txt

### 3. Set up your API key
Create a .env file in the root folder:
GROQ_API_KEY=your_groq_api_key_here

Get your free API key at: https://console.groq.com

### 4. Run the chatbot
Open src/chatbot.ipynb in VS Code and run all cells!

## Sample Conversation
**User:** Can I return my phone bought 5 days ago?

**Bot:** Since the phone is an electronic item and it's 
only been 5 days since you bought it, you can return it 
as long as it's unused and in its original packaging.

**User:** What if I already opened the box?

**Bot:** Unfortunately, electronic items must be unused 
and in original packaging to qualify for return.

## Key Concepts Demonstrated
- Retrieval Augmented Generation (RAG)
- Vector embeddings and semantic search
- Conversation memory management
- Secure API key handling
- Document chunking strategies
- Production-ready project structure

## What I Learned
Built this project while learning GenAI Data Engineering 
from scratch — covering LLMs, tokens, embeddings, 
vector databases, RAG pipelines, and conversational AI.

---
Built with dedication by [Hemanth Naidu Latchireddi] | Bengaluru, India
