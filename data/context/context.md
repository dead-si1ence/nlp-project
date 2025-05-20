# AI-Powered Local Document QA Desktop App

A local desktop application that allows you to upload files (like PDFs or text documents), extract and embed the content, and then ask questions about it using a fast, local AI model.

## Overview

This application provides a powerful, local document question-answering system that works entirely on your device, ensuring privacy and performance when analyzing documents. Upload your documents (PDFs, text files, etc.), and ask questions in natural language to get accurate answers based on the document content.

## Project Structure

- `ai_engine/`: Python code for AI document processing and QA
  - `extract.py`: Extracts text from documents like PDFs
  - `chunk.py`: Splits text into manageable chunks
  - `embed.py`: Generates vector embeddings of text chunks
  - `index.py`: Stores and retrieves embeddings using FAISS
  - `qa.py`: Answers questions using transformer models
  - `query.py`: Handles the query process
  - `cli.py`: Command line interface

## Features

- **Document Upload**: Process PDF and text documents
- **Text Extraction**: Extract content using PyMuPDF
- **Semantic Search**: Find relevant document sections based on questions
- **Local AI**: Uses transformer models for embedding and question answering
- **Vector Database**: Uses FAISS for efficient similarity search
- **Command Line Interface**: Simple interface for document processing and Q&A

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ai-doc-qa-app.git
   cd ai-doc-qa-app
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

Upload a document:

```bash
python -m ai_engine.cli upload path/to/document.pdf
```

Ask a question about your documents:

```bash
python -m ai_engine.cli ask "What is the main topic of the document?"
```

List all processed documents:

```bash
python -m ai_engine.cli list
```

Reset the document index:

```bash
python -m ai_engine.cli reset
```

## Core Technologies

- **PDF Processing**: PyMuPDF
- **Text Chunking**: NLTK
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Question Answering**: Hugging Face Transformers (deepset/roberta-base-squad2)

## License

[MIT License](LICENSE)

## Dataset Reference

- [Arabic Dialects Question and Answer Dataset](https://huggingface.co/datasets/CNTXTAI0/arabic_dialects_question_and_answer)
