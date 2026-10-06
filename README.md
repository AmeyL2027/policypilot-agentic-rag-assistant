# PolicyPilot

A local RAG assistant for querying policy documents with source-grounded answers.

## Overview

PolicyPilot is an end-to-end Retrieval-Augmented Generation (RAG) project built to answer questions over policy documents using a fully local and low-cost stack. The project ingests public and synthetic policy documents, chunks and indexes them into a vector database, retrieves relevant context for a user query, and generates grounded answers with visible source support.

The project also includes deterministic comparison logic for versioned travel policies, which improves reliability for change-analysis questions and reduces hallucination risk.

## Features

- Ingests public PDF documents and synthetic markdown policy documents
- Extracts raw text from documents
- Chunks documents for retrieval
- Generates local embeddings with Ollama
- Indexes chunks into local Qdrant
- Performs semantic retrieval over indexed chunks
- Generates source-grounded answers in a Streamlit UI
- Supports:
  - general policy questions
  - latest/current policy questions
  - deterministic version comparison for travel policy documents
- Runs fully locally without paid API usage

## Demo Questions

Try these in the app:

- What is the travel reimbursement policy?
- What is the latest travel reimbursement policy?
- What changed between Travel Expense Policy v1 and v2?
- What are the meal reimbursement limits?
- When are receipts required?

## Architecture

```text
Documents (PDF + Markdown)
        |
        v
Document Ingestion
        |
        v
Chunking Pipeline
        |
        v
Local Embeddings (Ollama: nomic-embed-text)
        |
        v
Vector Index (Qdrant local)
        |
        v
Retriever
        |
        +--> Deterministic version-comparison logic for travel policies
        |
        v
Answer Generation (Ollama: qwen2.5:1.5b)
        |
        v
Streamlit UI
