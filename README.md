# RAG Tutorial 1: Retrieval-Augmented Generation System

A comprehensive implementation of a Retrieval-Augmented Generation (RAG) system using LangChain, Google Generative AI, and Chroma vector database.

## 🎯 Project Overview

This project demonstrates a complete RAG pipeline with three main components:

1. **Document Ingestion Pipeline** - Load and process PDF documents, split them into chunks, and create embeddings
2. **Retrieval Pipeline** - Query the vector database to find relevant documents
3. **History-Aware Generation** - Generate responses using conversation history for context-aware answers

## 🚀 Features

- **PDF Document Loading**: Automatically loads and processes PDF files from the docs directory
- **Intelligent Text Chunking**: Uses recursive character splitting with configurable chunk sizes
- **Vector Embeddings**: Leverages Hugging Face embeddings (all-MiniLM-L6-v2 model)
- **Vector Database**: Chroma for efficient similarity search and retrieval
- **LLM Integration**: Google Generative AI (Gemini 2.5 Flash) for natural language generation
- **Conversation Memory**: Maintains chat history with context-aware query reformulation
- **Retriever Pattern**: LangChain's retriever interface for flexible document retrieval

## 📋 Prerequisites

- Python 3.13 or higher
- Google API Key (for Gemini API access)
- Optional: Groq API Key (for alternative LLM provider)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rag-tutorial1.git
   cd rag-tutorial1
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```
   Or using requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   GROQ_API_KEY=your_groq_api_key_here (optional)
   ```

## 📁 Project Structure

```
.
├── 1_ingestion_pipeline.py      # Document loading and embedding creation
├── 2_retrieval_pipeline.py      # Query and retrieval demonstration
├── 3_history_aware_generation.py # Conversation with history management
├── main.py                       # Main entry point
├── db/
│   └── chroma_db/               # Vector database storage (git-ignored)
├── docs/                        # PDF documents directory
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project configuration
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🎓 Usage

### 1. Prepare Your Documents

Place PDF files in the `docs/` directory:
```bash
mkdir docs
# Copy your PDF files here
```

### 2. Run the Ingestion Pipeline

This creates embeddings and stores them in the vector database:
```bash
python 1_ingestion_pipeline.py
```

### 3. Test Retrieval

Query the database for relevant documents:
```bash
python 2_retrieval_pipeline.py
```

### 4. Interactive Chat with History

Start a conversational interface with memory:
```bash
python 3_history_aware_generation.py
```

### 5. Run the Main Application

```bash
python main.py
```

## 🔑 Environment Variables

Create a `.env` file with the following variables:

```
GOOGLE_API_KEY=your_api_key_here
GROQ_API_KEY=your_api_key_here
```

See `.env.example` for reference.

## 📚 Components Explained

### Document Ingestion (`1_ingestion_pipeline.py`)
- Loads PDF files using `PyPDFLoader`
- Splits documents into chunks using `RecursiveCharacterTextSplitter`
- Creates embeddings using Hugging Face's `all-MiniLM-L6-v2` model
- Stores in Chroma vector database for persistence

### Retrieval (`2_retrieval_pipeline.py`)
- Connects to the persisted Chroma database
- Retrieves top-K relevant documents for a query
- Formats retrieved content for LLM input
- Generates answers using Google Generative AI

### Conversation History (`3_history_aware_generation.py`)
- Maintains chat history for context
- Reformulates questions based on conversation history
- Implements sliding window for history management
- Provides context-aware responses

## ⚙️ Configuration

### Embedding Model
Default: `sentence-transformers/all-MiniLM-L6-v2`

To use a different model, modify the model name in the pipeline files.

### LLM Model
Default: `gemini-2.5-flash`

Supported alternatives in the code:
- `gemini-pro`
- `grok-3` (via Groq API)
- Any other LLM supported by LangChain

### Chunk Configuration
- **Chunk Size**: 1000 characters (default)
- **Chunk Overlap**: 200 characters (default)

Adjust these in `1_ingestion_pipeline.py` for different use cases.

## 🐛 Troubleshooting

### No PDFs Found
- Ensure PDFs are in the `docs/` directory
- Check file extensions are `.pdf` (lowercase)

### API Key Errors
- Verify API keys are correctly set in `.env`
- Ensure `.env` file is in the project root

### Database Errors
- Delete `db/chroma_db/` and re-run ingestion pipeline
- Ensure write permissions in the project directory
