from fastapi import FastAPI
from app.routes import speech_recognition

app = FastAPI()

app.include_router(speech_recognition.router)