# Document Intelligence Platform

The Document Intelligence Platform is a professional research tool that uses Retrieval-Augmented Generation (RAG) to analyze PDF documents. It creates a local vector database of uploaded content to provide accurate, context-aware answers to natural language queries.
URL: https://document-summarise.streamlit.app/

## System Overview

The application follows a structured pipeline to transform static documents into searchable intelligence:

1. Data Ingestion: The system reads PDF files and verifies page counts. To ensure sufficient depth for analysis, only documents with more than 5 pages are processed into the knowledge base.
2. Text Processing: Content is split into smaller, overlapping chunks. This preservation of context ensures that the relationships between sentences remain intact during retrieval.
3. Vector Indexing: Using Google text-embedding models, the system converts text chunks into mathematical vectors stored in a FAISS database.
4. Intelligent Retrieval: When a query is submitted, the system performs a similarity search to find the most relevant document passages.
5. Response Generation: The Gemma 3 model synthesizes the retrieved context into a professional answer, complete with source citations.

## Technical Requirements

- Google Gemini API Key: Required for both text embeddings and large language model inference.
- Document Specifications: PDF format required.

## Usage Instructions

1. Upload one or more PDF documents.
2. Wait for the status indicator to confirm the Vector Database is ready.
3. Enter a question in the chat input at the bottom of the screen.
4. Review the generated response and the associated source citations.
