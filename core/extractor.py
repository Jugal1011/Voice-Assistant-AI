# Actionable Items, Decision, Questions 

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import os



def get_llm():
    return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.2)

def split_transcript(transcript: str) -> list:
    """Split the transcript into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return text_splitter.split_text(transcript) 

def extract_with_chunking(
    transcript: str,
    map_system_prompt: str,
    reduce_system_prompt: str,
) -> str:
    """Generic map-reduce extraction over transcript chunks."""
    llm = get_llm()

    # Map step
    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", map_system_prompt),
            ("human", "{transcript_chunk}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    partial_results = [
        map_chain.invoke({"transcript_chunk": chunk})
        for chunk in chunks
    ]

    combined_results = "\n\n".join(partial_results)

    # Reduce step
    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", reduce_system_prompt),
            ("human", "{results}"),
        ]
    )

    reduce_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"results": x})
        | reduce_prompt
        | llm
        | StrOutputParser()
    )

    return reduce_chain.invoke(combined_results)

def extract_action_items(transcript: str) -> str:
    return extract_with_chunking(
        transcript=transcript,
        map_system_prompt=(
            "You are an expert meeting analyst. "
            "Extract all action items from this transcript chunk. "
            "For each action item provide:\n"
            "- Task description\n"
            "- Owner (if mentioned)\n"
            "- Deadline (if mentioned, otherwise 'Not specified').\n"
            "If no action items exist, return 'No action items found in this chunk.'"
        ),
        reduce_system_prompt=(
            "You are an expert meeting analyst. "
            "Combine the extracted action items from multiple transcript chunks. "
            "Remove duplicates, merge similar items, preserve owners and deadlines, "
            "and return a clean numbered list. "
            "If no action items exist, return 'No action items found.'"
        ),
    )


def extract_key_decisions(transcript: str) -> str:
    return extract_with_chunking(
        transcript=transcript,
        map_system_prompt=(
            "You are an expert meeting analyst. "
            "Extract all key decisions made in this transcript chunk. "
            "If none exist, return 'No key decisions found in this chunk.'"
        ),
        reduce_system_prompt=(
            "Combine the key decisions extracted from multiple transcript chunks. "
            "Remove duplicates and merge similar decisions into a concise numbered list. "
            "If no decisions exist, return 'No key decisions found.'"
        ),
    )

def extract_questions(transcript: str) -> str:
    return extract_with_chunking(
        transcript=transcript,
        map_system_prompt=(
            "Extract all unresolved questions, pending discussions, "
            "or follow-up topics from this transcript chunk. "
            "If none exist, return 'No open questions found in this chunk.'"
        ),
        reduce_system_prompt=(
            "Combine the unresolved questions extracted from multiple transcript chunks. "
            "Remove duplicates and merge similar questions into a concise numbered list. "
            "If none exist, return 'No open questions found.'"
        ),
    )