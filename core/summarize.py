from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


import os

def get_llm():
    """Mistral LLM for summarization."""
    return ChatMistralAI(model="mistral-small-latest", temperature=0.3)

def split_transcript(transcript: str) -> list:
    """Split the transcript into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return text_splitter.split_text(transcript)

def summarize_transcript(transcript: str) -> str:
    """Summarize the transcript using Mistral LLM."""
    llm = get_llm()
    map_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize this portion of a meeting transcript concisely, highlighting key points and decisions made.",
            ),
            (
                "human",
                "{transcript_chunk}",
            ),
        ]
    )
    
    map_chain = map_prompt_template | llm | StrOutputParser()
    chunks = split_transcript(transcript)
    chunks_summaries = [map_chain.invoke({"transcript_chunk": chunk}) for chunk in chunks]
    combined_summary = "\n\n".join(chunks_summaries)
    
    combined_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. Combine the following partial summaries into a final professional meeting summary in bullet points.",
            ),
            (
                "human",
                "{summaries}",
            ),
        ]
    )
    
    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"summaries": x}) | combined_prompt_template | llm | StrOutputParser()
    )
    
    return combined_chain.invoke(combined_summary)


def generate_title(transcript: str) -> str:
    """Generate a title for the transcript using Mistral LLM."""
    llm = get_llm()
    title_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title (max 8 words). Only return the title, nothing else.",
            ),
            (
                "human",
                "{transcript}",
            ),
        ]
    )
    
    title_chain = (RunnablePassthrough() | RunnableLambda(lambda x: {"transcript": x}) | title_prompt_template | llm | StrOutputParser())
    return title_chain.invoke({"transcript": transcript[:2000]})