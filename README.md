# 🏥 Healthcare AI Chatbot

## Project Overview
Developed an AI-powered Healthcare Chatbot that enables users to upload healthcare PDF documents and ask questions based on the document content. The application uses Retrieval-Augmented Generation (RAG) and the Hugging Face Inference API to provide context-aware responses.

## Objectives
- Extract text from healthcare documents.
- Retrieve relevant information from uploaded PDFs.
- Generate accurate AI-based responses.
- Provide a simple and user-friendly interface.

## Features
- PDF document upload
- Text extraction
- Text chunking
- Embedding generation
- FAISS vector database
- Context retrieval (RAG)
- AI-powered question answering
- Interactive Streamlit web application

## Technologies Used
- Python
- Streamlit
- Hugging Face Inference API
- LangChain
- Sentence Transformers
- FAISS
- PyPDF
- NumPy

## Project Workflow
1. Upload Healthcare PDF
2. Extract Text
3. Split Text into Chunks
4. Generate Embeddings
5. Store Embeddings in FAISS
6. Retrieve Relevant Context
7. Generate AI Response
8. Display Answer

## Folder Structure
- `app.py` – Main application
- `hf_chat.py` – AI response generation
- `rag/` – Embeddings, Vector Store, Retriever
- `utils/` – PDF and text processing
- `requirements.txt` – Project dependencies

## Outcome
Successfully developed and deployed an AI Healthcare Chatbot capable of answering user queries based on uploaded healthcare documents using Retrieval-Augmented Generation (RAG).

# Healthcare-chatbot
