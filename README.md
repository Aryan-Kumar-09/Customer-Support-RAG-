# 🤖 AI Customer Support Bot (RAG)

This project is a conversational AI Customer Support Bot built using **Retrieval-Augmented Generation (RAG)**. It uses a custom knowledge base to accurately answer customer queries about shipping, returns, and technical support without hallucinating information.

The bot features a **History-Aware Retriever**, meaning it remembers the context of the conversation and can perfectly understand follow-up questions.

## 🚀 Features

* **RAG Architecture:** Grounds the AI's answers in a provided knowledge base (a mock FAQ document).
* **Conversational Memory:** Remembers previous questions and answers in the current session.
* **Local Vector Storage:** Uses FAISS to quickly search the document for relevant context.
* **OpenAI Integration:** Powered by `gpt-3.5-turbo` and `text-embedding-3-small`.

## 🛠️ Tech Stack

* **Python 3.8+**
* **LangChain** (Orchestration framework)
* **OpenAI API** (LLM & Embeddings)
* **FAISS** (Local Vector Database)

## 💻 Setup & Installation

**1. Clone the repository:**
```bash
git clone https://github.com/Aryan-Kumar-09/Customer-Support-RAG-
```
**2. Install Dependencies:**
```bash
pip install -r requirements.txt
```
**3. Add your API Key**
Export your OpenAI API key as an environment variable in your terminal:
```bash
export OPENAI_API_KEY="your-api-key-here"
```
**4. Run the Bot:**
```bash
python support.py
```
