from google.cloud import speech
import io

def transcribe_audio(audiofile):
    try:
        client = speech.SpeechClient()

        # Read the file contents
        file_contents = audiofile

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
    
    except Exception as e:
        raise ValueError(status_code=500, detail=str(e))
