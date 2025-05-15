# RAG with pgvector: Q&A

A simple question-answering system that uses RAG (Retrieval-Augmented Generation) to answer questions about given text file.

## Technologies Used

- **Python** with key libraries:
  - LangChain for RAG pipeline
  - OpenAI for embeddings and language models
  - PostgreSQL with pgvector for vector storage
  - Gradio for web interface

## How It Works

1. The text is split into chunks
2. Each chunk is converted into embeddings using OpenAI
3. Embeddings are stored in PostgreSQL with pgvector
4. When a question is asked:
   - System finds the most relevant text chunks
   - Uses these chunks as context to generate an answer
   - Returns the answer through a chat interface

