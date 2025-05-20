
# AI-Powered Local Document QA Desktop App - Full Project Roadmap

This document outlines every step of building a local desktop application that allows you to upload files (like PDFs or text documents), extract and embed the content, and then ask questions about it using a fast, local AI model.

---

## Project Summary

**Frontend:** React + TypeScript + Tailwind CSS inside Tauri  
**Backend:** Rust (Tauri backend for native bridge + command runner)  
**AI Logic:** Python (document processing, embedding, question answering)

---

## 1. Project Setup

### 1.1. Directory Structure

```text
ai-doc-qa-app/
├── frontend/         # React + TS frontend
├── backend/          # Rust backend (Tauri)
├── ai_engine/        # Python code for processing and answering questions
├── docs/             # Documentation and design notes
└── README.md
```

### 1.2. Prerequisites

Install the following:

- [Node.js](https://nodejs.org/) (v18+)
- [Rust](https://www.rust-lang.org/)
- [Tauri CLI](https://tauri.app/v1/guides/getting-started/setup/)
- Python 3.9+
- Virtual environment manager: `venv`, `poetry`, or `conda`
- `pip install -U pip`

---

## 2. Frontend (Tauri + React + TS)

### 2.1. Create the frontend app

```bash
cd ai-doc-qa-app
npx create-react-app frontend --template typescript
cd frontend
npm install tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

Update `tailwind.config.js` and CSS file for Tailwind.

### 2.2. Setup File Upload & UI

Install:

```bash
npm install react-dropzone axios @tauri-apps/api react-loader-spinner
```

Implement:

- File drop/upload component using `react-dropzone`
- File metadata list and selection
- Simple input box for asking questions
- Loading spinner while answer is loading

---

## 3. Backend (Tauri + Rust)

### 3.1. Initialize Tauri

```bash
cd ..
cargo install create-tauri-app
create-tauri-app backend
cd backend
```

Modify Tauri config to point to `../frontend/dist` as the UI.

### 3.2. Add Rust APIs

Use Tauri `command` interface to:

- Receive files (paths) and pass to Python for processing
- Run Python scripts with `std::process::Command`
- Return JSON back to frontend

Use crates:

```toml
[dependencies]
tauri = { version = "1", features = ["api-all"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
uuid = "1"
tokio = { version = "1", features = ["full"] }
```

---

## 4. AI Engine (Python)

### 4.1. Set Up Python Env

```bash
cd ai_engine
python -m venv venv
source venv/bin/activate
pip install torch transformers sentence-transformers faiss-cpu PyMuPDF nltk
```

Optionally install:

```bash
pip install pdfplumber python-docx typer
```

### 4.2. Modules

- `extract.py`: Extract text from PDFs using `PyMuPDF`
- `chunk.py`: Chunk text by paragraphs or sentence groups (use NLTK or regex)
- `embed.py`: Use `all-MiniLM-L6-v2` from `sentence-transformers` to embed
- `index.py`: Store embeddings in `FAISS` index and save with `pickle`
- `query.py`: Take question, embed, retrieve top-K from index, pass to QA model
- `qa.py`: Use `deepset/roberta-base-squad2` with `transformers` pipeline

### 4.3. Running the pipeline

On file upload:

1. Extract text
2. Chunk and embed
3. Store in local FAISS index
4. Map chunks to metadata (filename, page, chunk_id)

On question:

1. Embed question
2. Retrieve top-K similar chunks
3. Pass to Roberta QA model and return answer

---

## 5. Sample Script to Process Files

```bash
python cli.py --upload my_file.pdf
python cli.py --ask "What is a neural network?"
```

---

## 6. Communication Between Rust and Python

Use `Command::new("python3")` to call `cli.py` with args like `--ask`.

Capture `stdout`, parse JSON, and send it back to frontend.

---

## 7. Caching and Memory Handling

- Store FAISS index and embeddings in memory (load once per session)
- Optionally cache answers per question in `sqlite3` or `json`

---

## 8. Testing

- Unit test each Python module
- Integration test full pipeline on dummy PDFs
- Test with multiple PDFs and large queries

---

## 9. Deployment

- Bundle app using `tauri build`
- Test on Linux, Windows, macOS
- Package using `.deb`, `.msi`, or `.dmg`

---

## 10. Future Features (Optional)

- Summarize files
- Highlight source in answers
- Natural multi-turn questions
- Model switch (TinyBERT, ONNX-optimized models)
- Full chapter QA mode (e.g., ask across all files)

---

## Done

You're ready to build a fast, local, AI-powered question answering desktop app for your study notes or technical files.
