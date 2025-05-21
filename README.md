# AI-Powered Local Document QA Desktop App

A local desktop application that allows you to upload files (like PDFs or text documents), extract and embed the content, and then ask questions about it using a fast, local AI model.

## Overview

This application provides a powerful, local document question-answering system that works entirely on your device, ensuring privacy and performance when analyzing documents. Upload your documents (PDFs, text files, etc.), and ask questions in natural language to get accurate answers based on the document content.

## Key Benefits

- **Privacy-First**: All processing happens locally - no data is sent to external servers
- **Fast and Efficient**: Optimized vector search delivers quick responses to your questions
- **Works Offline**: No internet connection required after initial setup
- **Supports Multiple Documents**: Ask questions that span across all your uploaded documents
- **Source Attribution**: Answers include references to the source documents and page numbers

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
- **Multi-document Support**: Search across all uploaded documents simultaneously
- **Context-aware Answers**: Responses are generated from the most relevant document sections
- **Source References**: Answers include references to source documents and page numbers
- **Customizable Parameters**: Adjust chunking, embedding, and retrieval settings

## System Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+ recommended)
- **CPU**: Intel Core i5/AMD Ryzen 5 or better
- **RAM**: 8GB minimum, 16GB recommended for larger documents
- **Storage**: 2GB for application and models, plus space for your documents
- **Python**: Version 3.9 or higher

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/dead-si1ence/nlp-project.git
   cd nlp-project
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

4. Download required NLTK data (required for text chunking):

   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

5. Create necessary directories:

   ```bash
   mkdir -p data/index
   ```

## Usage

### Command Line Interface

The application provides a simple command-line interface for document management and querying.

#### Upload a Document

Process and index a PDF, text, or Microsoft Word document:

```bash
python -m ai_engine.cli upload path/to/document.pdf
```

The system will:

1. Extract text from the document
2. Split the text into semantic chunks
3. Generate embeddings for each chunk
4. Store the embeddings in the vector index
5. Return a success confirmation with document ID

#### Ask Questions

Query your documents with natural language questions:

```bash
python -m ai_engine.cli ask "What is the main topic of the document?"
```

Optional parameters:

- `--top-k 3` (retrieve top 3 most relevant chunks, default is 5)

```bash
python -m ai_engine.cli ask "Explain the concept of vector embeddings" --top-k 3
```

#### List Documents

View all documents currently indexed in the system:

```bash
python -m ai_engine.cli list
```

This displays:

- Document IDs
- File paths
- File names
- Number of chunks per document

#### Reset the Index

Remove all documents from the index:

```bash
python -m ai_engine.cli reset
```

This operation cannot be undone, so use with caution!

### Example Workflow

```bash
# Activate your virtual environment
source venv/bin/activate

# Upload some documents
python -m ai_engine.cli upload research_papers/quantum_computing.pdf
python -m ai_engine.cli upload textbooks/machine_learning_basics.pdf

# Check what documents are in the index
python -m ai_engine.cli list

# Ask questions about your documents
python -m ai_engine.cli ask "What are the key differences between quantum and classical computing?"
python -m ai_engine.cli ask "Explain gradient descent in simple terms"

# When you're done, you can reset the index if needed
# python -m ai_engine.cli reset
```

## Customization

You can customize the application by modifying the following parameters in the source code:

### Embedding Model

The default embedding model is `all-MiniLM-L6-v2`, which provides a good balance between performance and accuracy. If you want to use a different model, modify the `embedderModel` parameter in the `DocumentQA` class in `ai_engine/cli.py`.

### QA Model

The default QA model is `deepset/roberta-base-squad2`. You can change this to any Hugging Face model that supports extractive QA by modifying the `qaModel` parameter in the `DocumentQA` class.

### Chunking Parameters

Text chunking can be customized in the `TextChunker` class in `ai_engine/chunk.py`:

- `chunkSize`: The maximum size of each text chunk (default: 512)
- `chunkOverlap`: The overlap between consecutive chunks (default: 50)

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError" on first run**:
   - Make sure you have activated the virtual environment
   - Verify all dependencies are installed: `pip install -r requirements.txt`

2. **Slow performance on large documents**:
   - Consider increasing chunk size to reduce the number of embeddings
   - Running on a machine with more RAM can improve performance

3. **Models downloading repeatedly**:
   - Models are cached in the `~/.cache/huggingface` directory
   - Check file permissions if models are downloading on each run

4. **Out of memory errors**:
   - Process fewer documents at once
   - Reduce the chunk size in the `TextChunker` class

## Core Technologies

- **PDF Processing**: PyMuPDF
- **Text Chunking**: NLTK
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Question Answering**: Hugging Face Transformers (deepset/roberta-base-squad2)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

[MIT License](LICENSE)
