from fastapi import FastAPI
from routes import websocket

app = FastAPI()

app.include_router(websocket.router)