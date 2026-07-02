from dotenv import load_dotenv
load_dotenv()
from utils.audio_preprocessor import process_input_audio
from core.transcriber import transcribe_audio_chunks
import os

print("Key loaded ", os.getenv('SARVAM_API_KEY'))
print(os.getcwd())
source = "https://www.youtube.com/watch?v=ocRzt5NvI7A"
chunks = process_input_audio(source)
transcription = transcribe_audio_chunks(chunks, language="hinglish")
print("Final Transcription:")
print(transcription)