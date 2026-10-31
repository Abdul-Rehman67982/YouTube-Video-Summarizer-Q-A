
# ---------------- Imports ----------------
import os
import re
from functools import lru_cache

import gradio as gr
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # Load GROQ_API_KEY from the .env file

GROQ_MODEL = "qwen/qwen3.8-27b"


# ---------------- Transcript Extraction ----------------
def get_video_id(url):
    pattern = r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_transcript(url):
    video_id = get_video_id(url)

    if not video_id:
        return None

    ytt_api = YouTubeTranscriptApi()

    try:
        transcripts = ytt_api.list(video_id)
    except Exception as e:
        print(f"Failed to retrieve transcript list: {e}")
        return None

    transcript = ""

    for t in transcripts:
        if t.language_code == 'en':
            if t.is_generated:
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                transcript = t.fetch()
                break

    return transcript if transcript else None


def process(transcript):
    text = ""

    if not transcript:
        return text

    for i in transcript:
        try:
            text += f"Text: {i.text} Start: {i.start}\n"
        except AttributeError:
            pass

    return text


def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return text_splitter.split_text(processed_transcript)


# ---------------- Groq LLM and Embeddings ----------------
@lru_cache(maxsize=1)
def setup_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=900
    )


@lru_cache(maxsize=1)
def setup_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_faiss_index(chunks, embedding_model):
    return FAISS.from_texts(chunks, embedding_model)


def perform_similarity_search(faiss_index, query, k=3):
    return faiss_index.similarity_search(query, k=k)


# ---------------- Video Summarization ----------------
def create_summary_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are an AI assistant tasked with summarizing YouTube video transcripts. "
         "Provide concise, informative summaries that capture the main points of the video content.\n\n"
         "Instructions:\n"
         "1. Summarize the transcript in a single concise paragraph.\n"
         "2. Ignore any timestamps in your summary.\n"
         "3. Focus on the spoken content (Text) of the video.\n\n"
         'Note: In the transcript, "Text" refers to the spoken words in the video, '
         'and "start" indicates the timestamp when that part begins in the video.'),
        ("human",
         "Please summarize the following YouTube video transcript:\n\n{transcript}"),
    ])


def create_summary_chain(llm, prompt):
    return prompt | llm | StrOutputParser()


# ---------------- Question Answering (RAG) ----------------
def retrieve(query, faiss_index, k=7):
    return faiss_index.similarity_search(query, k=k)


def create_qa_prompt_template():
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert assistant providing detailed and accurate answers based on the following video content. "
         "Your responses should be:\n"
         "1. Precise and free from repetition\n"
         "2. Consistent with the information provided in the video\n"
         "3. Well-organized and easy to understand\n"
         "4. Focused on addressing the user's question directly\n"
         "If you encounter conflicting information in the video content, use your best judgment "
         "to provide the most likely correct answer based on context.\n"
         'Note: In the transcript, "Text" refers to the spoken words in the video, '
         'and "start" indicates the timestamp when that part begins in the video.'),
        ("human",
         "Relevant Video Context:\n{context}\n\n"
         "Based on the above context, please answer the following question:\n{question}"),
    ])


def create_qa_chain(llm, prompt_template):
    return prompt_template | llm | StrOutputParser()


def generate_answer(question, faiss_index, qa_chain, k=7):
    relevant_docs = retrieve(question, faiss_index, k=k)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    return qa_chain.invoke({
        "context": context,
        "question": question
    })


# ---------------- Application Logic ----------------
current_url = ""
processed_transcript = ""


def load_transcript(video_url):
    global current_url, processed_transcript

    if video_url != current_url or not processed_transcript:
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
        current_url = video_url

    return processed_transcript


def summarize_video(video_url):
    if not video_url:
        return "Please provide a valid YouTube URL."

    transcript_text = load_transcript(video_url)

    if not transcript_text:
        return "No transcript available. Please check the URL or try another video."

    try:
        summary_chain = create_summary_chain(
            setup_llm(),
            create_summary_prompt()
        )

        return summary_chain.invoke({
            "transcript": transcript_text
        })

    except Exception as e:
        return f"Error: {e}"


def answer_question(video_url, user_question):
    if not video_url:
        return "Please provide a valid YouTube URL."

    if not user_question:
        return "Please provide a valid question."

    transcript_text = load_transcript(video_url)

    if not transcript_text:
        return "No transcript available. Please check the URL or try another video."

    try:
        chunks = chunk_transcript(transcript_text)

        llm = setup_llm()
        embedding_model = setup_embedding_model()

        faiss_index = create_faiss_index(chunks, embedding_model)

        qa_chain = create_qa_chain(
            llm,
            create_qa_prompt_template()
        )

        return generate_answer(
            user_question,
            faiss_index,
            qa_chain
        )

    except Exception as e:
        return f"Error: {e}"


# ---------------- Gradio User Interface ----------------
with gr.Blocks() as interface:

    gr.Markdown(
        "<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A (Groq)</h2>"
    )

    video_url = gr.Textbox(
        label="YouTube Video URL",
        placeholder="Enter the YouTube Video URL"
    )

    summary_output = gr.Textbox(
        label="Video Summary",
        lines=5
    )

    question_input = gr.Textbox(
        label="Ask a Question About the Video",
        placeholder="Ask your question"
    )

    answer_output = gr.Textbox(
        label="Answer to Your Question",
        lines=5
    )

    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")

    summarize_btn.click(
        summarize_video,
        inputs=video_url,
        outputs=summary_output
    )

    question_btn.click(
        answer_question,
        inputs=[video_url, question_input],
        outputs=answer_output
    )


# Launch the application at http://127.0.0.1:7860
interface.launch()
