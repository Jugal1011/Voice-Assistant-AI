import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download__youtube_audio(url : str) -> str:
    """Extract the audio file from a YouTube video and save it as a WAV file."""
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best', 
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        audio_file_path = ydl.prepare_filename(info_dict)
        audio_file_path = os.path.splitext(audio_file_path)[0] + ".wav"
        return audio_file_path 
    
# data = download__youtube_audio("https://www.youtube.com/watch?v=tD0F5CiuHek")

def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16kHz
    audio.export(output_path, format="wav")
    return output_path

# data_final = convert_to_wav(data)

def chunk_audio(wav_path: str, chunk_mins:int = 10) -> list:
    """Chunk a WAV audio file into smaller segments of specified length in minutes."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_length_ms = chunk_mins * 60 * 1000  # Convert minutes to milliseconds
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
        end = min(start + chunk_length_ms, len(audio))
        chunk = audio[start:end]
        chunk_filename = f"{os.path.splitext(wav_path)[0]}_chunk_{i}.wav"
        chunk.export(chunk_filename, format="wav")
        chunks.append(chunk_filename)
    return chunks

# print(chunk_audio(data_final, chunk_mins=10))

def process_input_audio(source: str) -> list:
    if source.startswith("http") or source.startswith("https"):
        print("Downloading audio from YouTube...")
        wav_path = download__youtube_audio(source)
    else:
        print("Converting local audio file to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunked_files = chunk_audio(wav_path,10)
    print(f"Audio processing complete. {len(chunked_files)} chunk(s) created.")
    return chunked_files