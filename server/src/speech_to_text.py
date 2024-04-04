from google.cloud import speech
import io

def transcribe_audio(audiofile, language='en-US'):

    client = speech.SpeechClient()

    with open(audiofile, 'rb') as audio_file:
        file_contents = audio_file.read()

    audio = speech.RecognitionAudio(content=file_contents)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language,
    )

    try:
        response = client.recognize(config=config, audio=audio)

        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript

        return {"transcript": transcript}

    except Exception as e:
        print(f"An Error occured during the transcription. Explanation: {e}")
        return {"transcript": "", "error": str(e)}