from fastapi import APIRouter, UploadFile, File
from google.cloud import speech
import io

router = APIRouter()

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    client = speech.SpeechClient()

    # Read the file contents
    file_contents = await file.read()

    audio = speech.RecognitionAudio(content=file_contents)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    return {"transcript": transcript}

