import os
import traceback
from openai import OpenAI

def transcribe_audio_to_text(file_path: str) -> str:
    print(f"Starting OpenAI Whisper transcription of: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    file_size = os.path.getsize(file_path)
    print(f"Audio file size: {file_size} bytes")
    
    if file_size == 0:
        raise ValueError("Audio file is empty")
    
    # Check if OpenAI API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Warning: OpenAI API key not found, returning placeholder text")
        return "Transcription not available - OpenAI API key not configured"
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        print("Sending audio to OpenAI Whisper API...")
        
        # Open and send audio file to Whisper API
        with open(file_path, 'rb') as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
                # Note: Omitting language parameter allows auto-detection
                # If you want to specify language, use ISO-639-1 format like "en", "es", "fr", etc.
            )
        
        result_text = response.text.strip()
        print(f"Transcription completed. Text length: {len(result_text)}")
        print(f"Transcription preview: {result_text[:100]}...")
        
        if not result_text:
            return "No speech detected in audio"
        
        return result_text
        
    except Exception as e:
        print(f"OpenAI Whisper transcription error: {e}")
        traceback.print_exc()
        # Return a descriptive error instead of raising
        return f"Transcription failed: {str(e)}"
