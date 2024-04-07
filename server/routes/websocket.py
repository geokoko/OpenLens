import openai
from fastapi import APIRouter, HTTPException, WebSocket
from src.speech_to_text import transcribe_audio
from src.video_analysis import VideoAnalysis

openai.api_key = ''

router = APIRouter()

@router.websocket("/ws/generate-response")
async def generate_response(websocket: WebSocket):
    await websocket.accept()
    while 1:
        data = await websocket.receive_text()
        
        prompt = f"Please continue this conversation when given these details: [Emotion: {emotion}]\n[Speech: {speech}]\n[Response: "
        try:
            response = openai.Completion.create(
                engine="gpt-3.5",
                prompt=prompt,
                max_tokens=50,
                n=1,
                stop=["\n", " [Response: "]
            )
            await websocket.send_text(response.choices[0].text.strip())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
