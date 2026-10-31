# YouTube Video Summarizer & Q&A

An AI-powered application that summarizes YouTube videos and answers questions based on their transcripts using Retrieval-Augmented Generation (RAG).

## Features

- **YouTube Transcript Extraction:** Extracts English transcripts from YouTube videos.
- **Text Processing:** Splits transcripts into smaller chunks using Recursive Character Text Splitter.
- **Text Embeddings:** Converts transcript chunks into vector embeddings using Hugging Face.
- **Vector Search:** Uses FAISS to retrieve relevant transcript chunks based on semantic similarity.
- **AI Summarization:** Generates concise video summaries using Groq LLM.
- **Question Answering:** Answers user questions using retrieved transcript context.
- **Interactive UI:** Provides a simple interface built with Gradio.

## Tech Stack

- Python
- LangChain
- Groq API
- Hugging Face Sentence Transformers
- FAISS
- YouTube Transcript API
- Gradio
- Python-dotenv

## How It Works

1. User provides a YouTube video URL.
2. The application extracts the video's English transcript.
3. The transcript is processed and divided into smaller chunks.
4. Hugging Face generates embeddings for each chunk.
5. FAISS creates a vector index for similarity search.
6. Groq LLM generates a summary or answers questions using the relevant retrieved context.

## Project Architecture

```text
YouTube Video URL
       |
       v
Transcript Extraction
       |
       v
Text Processing
       |
       v
Text Chunking
       |
       v
Hugging Face Embeddings
       |
       v
FAISS Vector Index
       |
       v
Similarity Search
       |
       v
Groq LLM
       |
       v
Summary / Question Answer
```

## Installation

**1. Clone the repository**

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

**2. Create and activate a virtual environment**

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install youtube-transcript-api faiss-cpu langchain langchain-community langchain-text-splitters langchain-groq langchain-huggingface sentence-transformers gradio python-dotenv
```

**4. Configure environment variables**

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

**5. Run the application**

```bash
python app.py
```

Open the local Gradio URL displayed in your terminal.

## Use Cases

- Quickly understand long YouTube videos.
- Generate summaries without watching the entire video.
- Ask specific questions about video content.
- Retrieve relevant information from video transcripts.

## Learning Outcomes

Through this project, I practiced:

- Building an end-to-end RAG pipeline.
- Working with text chunking and embeddings.
- Creating vector indexes using FAISS.
- Performing semantic similarity search.
- Integrating LLMs with LangChain.
- Building an interactive AI application with Gradio.

## Future Improvements

- Support for multiple languages.
- Better transcript and URL validation.
- Cache vector indexes to avoid rebuilding them for every question.
- Support for multiple videos and persistent storage.

---

**Built with Python, LangChain, FAISS, Hugging Face, and Groq.**
