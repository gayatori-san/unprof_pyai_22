# Conversational RAG CLI with Memory

## 📖 Overview

This project demonstrates how to build a **Conversational Retrieval-Augmented Generation (RAG)** system using **LangChain**, **OpenAI**, and **FAISS**.

Unlike a basic RAG application, this implementation maintains **conversation history**, allowing users to ask follow-up questions naturally. The system automatically reformulates context-dependent questions into standalone queries before retrieving relevant information from the knowledge base.

For demonstration purposes, the project uses a small in-memory document collection stored in a local FAISS vector database.

---

# 🚀 Features

* Conversational RAG pipeline
* Local FAISS vector database
* OpenAI Embeddings for semantic search
* History-aware question reformulation
* Automatic retrieval of relevant documents
* Memory-based conversations
* Command Line Interface (CLI)
* Concise and grounded responses
* Graceful error handling

---

# 🏗️ Project Structure

```text
.
├── task 22.py                 # Main Conversational RAG application
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies
```

---

# ⚙️ Architecture

```text
                 User Question
                      │
                      ▼
            Conversation History
                      │
                      ▼
        History-Aware Question Rewriter
                      │
                      ▼
         Standalone User Question
                      │
                      ▼
          FAISS Vector Retriever
                      │
                      ▼
          Relevant Document Chunks
                      │
                      ▼
          OpenAI Chat Language Model
                      │
                      ▼
               Final AI Response
                      │
                      ▼
      Update Conversation History
```

---

# 🔄 Workflow

### Step 1 — Build the Vector Store

The application creates a FAISS vector database from a small collection of documents using OpenAI embeddings.

```python
FAISS.from_texts(documents, embeddings)
```

---

### Step 2 — Create a History-Aware Retriever

LangChain first analyzes the chat history to rewrite follow-up questions into complete standalone questions.

Example:

**Conversation**

```text
User: What is Kubuntu?
AI: Kubuntu is an Ubuntu distribution using KDE Plasma.

User: Who created its desktop?
```

The retriever automatically reformulates it as:

```text
Who created the KDE Plasma Desktop used by Kubuntu?
```

This significantly improves retrieval quality.

---

### Step 3 — Retrieve Relevant Documents

The retriever performs semantic similarity search on the FAISS index and returns the most relevant document chunks.

Current configuration:

```python
search_kwargs={"k": 2}
```

Only the top two relevant chunks are retrieved.

---

### Step 4 — Generate the Answer

The retrieved context is inserted into a prompt and sent to the OpenAI Chat Model.

The assistant is instructed to:

* Use only retrieved context
* Avoid hallucinations
* Keep answers concise
* Say "I don't know" when context is insufficient

---

### Step 5 — Update Conversation Memory

After each interaction, both the user's message and the assistant's response are stored in memory.

```python
chat_history.extend([
    HumanMessage(content=query),
    AIMessage(content=answer)
])
```

This enables natural multi-turn conversations.

---

# 📦 Installation

## Clone the repository

```bash
git clone https://github.com/gayatori-san/unprof_pyai_22.git

cd conversational-rag-cli
```

---

## Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

or install manually

```bash
pip install langchain
pip install langchain-openai
pip install langchain-community
pip install faiss-cpu
pip install openai
pip install tiktoken
```

---

# 🔑 Configure OpenAI API Key

### Linux / macOS

```bash
export OPENAI_API_KEY="your_api_key"
```

### Windows Command Prompt

```cmd
set OPENAI_API_KEY=your_api_key
```

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

---

# ▶️ Running the Application

```bash
python app.py
```

---

# 💬 Example Conversation

```text
Welcome to the Conversational RAG CLI!

📝 You:
What is Kubuntu?

💬 AI:
Kubuntu is an official flavor of Ubuntu that uses the KDE Plasma Desktop instead of GNOME.

📝 You:
Who created its desktop?

💬 AI:
The KDE Plasma Desktop was created by Matthias Ettrich in 1996.

📝 You:
What does RAG stand for?

💬 AI:
RAG stands for Retrieval-Augmented Generation, combining search with Large Language Models.
```

---

# 🧠 Technologies Used

* Python 3
* LangChain
* OpenAI GPT-3.5 Turbo
* OpenAI Embeddings
* FAISS Vector Store
* Command Line Interface (CLI)

---

# 📚 Concepts Demonstrated

This project covers several important RAG concepts:

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Embeddings
* FAISS Vector Database
* Conversational Memory
* History-Aware Retrieval
* Prompt Engineering
* Multi-turn Question Answering
* LangChain Chains
* Document Retrieval Pipeline

---

# 🔮 Future Improvements

Possible enhancements include:

* Load documents from PDF files
* Automatic document chunking
* Persistent FAISS index
* Support for ChromaDB or Pinecone
* Streaming AI responses
* Source document citations
* Web interface using Streamlit or Flask
* Long-term conversation memory
* Multiple knowledge bases
* Hybrid search (keyword + semantic)

---

# 📄 Requirements

Example `requirements.txt`

```text
langchain
langchain-openai
langchain-community
faiss-cpu
openai
tiktoken
```

---

# 🎯 Learning Outcomes

After completing this project, you will understand how to:

* Build a conversational RAG system
* Create a FAISS vector database
* Generate semantic embeddings
* Use LangChain retrieval chains
* Implement history-aware retrieval
* Maintain conversation memory
* Develop multi-turn AI assistants
* Integrate OpenAI with LangChain
* Build end-to-end conversational AI applications

