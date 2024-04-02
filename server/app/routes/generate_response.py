import openai
from fastapi import APIRouter, HTTPException

openai.api_key = ''

router = APIRouter()

@router.post("/generate-response")
async def generate_response(emotion, speech):
    prompt = f"[Emotion: {emotion}]\n[Speech: {speech}]\n[Response: "
    try:
        response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=["\n", " [Response: "]
        )
        return {"response": response.choices[0].text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
