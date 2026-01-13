import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.video_analysis import VideoAnalysis
import json

router = APIRouter()

# Global init for performance
# Note: Ensure weights have been generated or downloaded
MODEL_PATH = "server/src/models/deep_emotion_best.pt"
video_analyzer = VideoAnalysis(model_path=MODEL_PATH)

OLLAMA_URL = "http://localhost:11434/api/generate"
# You can change this to "llama3.2", "gemma2", "mistral", etc. based on what you pulled.
LLM_MODEL = "llama3.2" 

@router.websocket("/ws/generate-response")
async def generate_response(websocket: WebSocket):
    await websocket.accept()
    print(f"Client connected. Using Local LLM: {LLM_MODEL}")
    
    last_emotion = "neutral"
    frame_count = 0
    client = httpx.AsyncClient()

    try:
        while True:
            # Receive binary frame from C client
            data = await websocket.receive_bytes()
            frame_count += 1
            
            # Analyze every 5th frame to save resources
            if frame_count % 5 == 0:
                emotion = video_analyzer.detect_emotion(data)
                
                # Logic: Only speak if emotion changes
                if emotion and emotion != last_emotion:
                    print(f"Detected Emotion Change: {emotion}")
                    last_emotion = emotion
                    
                    # Prompt for the Local LLM
                    prompt = f"The user is wearing smart glasses. You detect they look {emotion}. Give a very short (max 1 sentence), helpful, cool, or supportive comment to display on their HUD."
                    
                    payload = {
                        "model": LLM_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_ctx": 128, # Keep context small for speed
                            "temperature": 0.7
                        }
                    }

                    try:
                        # Call Ollama locally
                        response = await client.post(OLLAMA_URL, json=payload, timeout=2.0)
                        if response.status_code == 200:
                            result = response.json()
                            text_response = result.get("response", "").strip()
                            print(f"LLM says: {text_response}")
                            await websocket.send_text(text_response)
                        else:
                            print(f"Ollama Error {response.status_code}")
                    except httpx.ReadTimeout:
                        print("LLM Timed out (is your GPU busy?)")
                    except Exception as e:
                        print(f"LLM Connection Error: {e}")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        await client.aclose()
