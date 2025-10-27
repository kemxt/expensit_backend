import asyncio
import base64
import datetime
from io import BytesIO
import io
import os
import tempfile
import time
from PIL import Image
from fastapi import Body, FastAPI, HTTPException, APIRouter, requests
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from openai import AsyncOpenAI, OpenAI
import requests
import json
from dotenv import load_dotenv
from typing import Dict, Any
from pydantic import BaseModel, HttpUrl
import firebase_admin
from firebase_admin import credentials, storage, auth
from fastapi import Body, FastAPI, HTTPException, Depends, Header
from typing import Dict, Optional
from users import get_or_create_assistant, get_or_create_thread

router = APIRouter()

load_dotenv()

app = FastAPI()


class EmbeddingRequest(BaseModel):
    text: str

class ImageAnalysisRequest(BaseModel):
    image_url: HttpUrl
    


cred = credentials.Certificate("config/expensit-10546-firebase-adminsdk-q25nr-f97f281a02.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'gs://expensit-10546.appspot.com/images'
})


async def verify_firebase_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    token = authorization.split("Bearer ")[1]
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


@router.post("/analyze-image")
def analyze_image(
    image_data: dict = Body(...),
    user_token: dict = Depends(verify_firebase_token)
):
    try:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
        user_id = user_token["uid"]
        image_path = image_data["image_path"].strip().lstrip('/')
        
        # Weryfikacja dostępu
        if not image_path.startswith(f"users/{user_id}/"):
            return JSONResponse(status_code=403, content={"error": "Access denied"})

        # Pobranie signed URL
        bucket = storage.bucket('expensit-10546.appspot.com')
        blob = bucket.blob(image_path)
        
        if not blob.exists():
            return JSONResponse(status_code=404, content={"error": "Image not found"})
        
        signed_url = blob.generate_signed_url(expiration=datetime.timedelta(minutes=15))

        analysis_response = openai_client.chat.completions.create(
            model="gpt-4o",  
            messages=[
                {
                    "role": "system",
                    "content": os.getenv("GPT_PROMPT")  
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this receipt and return JSON with: shop, date, total, items[]"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": signed_url}
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"}  # Wymusza JSON
        )
        
        
        receipt_json = json.loads(analysis_response.choices[0].message.content)
        
       
        agent_id = get_or_create_assistant(user_id)
        thread_id = get_or_create_thread(user_id, agent_id)
        print(agent_id)
        print(thread_id)
        # Dodaj do thread'a tylko tekst z JSONem
        openai_client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=f"Receipt analyzed:\n```json\n{json.dumps(receipt_json, indent=2)}\n```"
        )
        
        # Opcjonalnie: Dodaj odpowiedź asystenta żeby zatwierdzić
        openai_client.beta.threads.messages.create(
            thread_id=thread_id,
            role="assistant",
            content=f"Receipt saved: {receipt_json.get('shop', 'Unknown')} - {receipt_json.get('total', 0)} PLN"
        )
        
        return {
            "status": "success",
            "analysis": receipt_json,
            "message": "Receipt analyzed and added to conversation history"
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@router.post("/chat")
async def chat(
    request: dict = Body(...),
    user_token: dict = Depends(verify_firebase_token)
):
    try:
        print("Start")
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        user_id = user_token["uid"]
        user_message = request.get("message", "").strip()
        
        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"error": "Message cannot be empty"}
            )
        
        # Pobierz asystenta i thread
        agent_id = get_or_create_assistant(user_id)
        thread_id = get_or_create_thread(user_id, agent_id)
        
        print(f"User: {user_id}")
        print(f"Agent: {agent_id}")
        print(f"Thread: {thread_id}")
        print(f"Message: {user_message}")
        
        # Dodaj wiadomość użytkownika do thread'a
        openai_client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_message
        )
        
        # Generator do streamowania odpowiedzi
        async def generate():
            try:
                with openai_client.beta.threads.runs.stream(
                    thread_id=thread_id,
                    assistant_id=agent_id,
                ) as stream:
                    for event in stream:
                        # Streamuj fragmenty tekstu
                        if event.event == "thread.message.delta":
                            for content in event.data.delta.content:
                                if hasattr(content, 'text') and hasattr(content.text, 'value'):
                                    yield f"data: {json.dumps({'type': 'content', 'text': content.text.value})}\n\n"
                        
                        # Wyślij status zakończenia
                        elif event.event == "thread.run.completed":
                            yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id})}\n\n"
                        
                        # Obsłuż błędy
                        elif event.event == "thread.run.failed":
                            yield f"data: {json.dumps({'type': 'error', 'message': 'Run failed'})}\n\n"
                            
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
            
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

@router.delete("/chat/clear")
def clear_chat_history(user_token: dict = Depends(verify_firebase_token)):
    try:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        user_id = user_token["uid"]
        
        # Pobierz obecny thread_id
        agent_id = get_or_create_assistant(user_id)
        old_thread_id = get_or_create_thread(user_id, agent_id)
        
        # Usuń stary thread
        try:
            openai_client.beta.threads.delete(old_thread_id)
            print(f"Deleted thread: {old_thread_id}")
        except Exception as e:
            print(f"Error deleting thread: {e}")
        
        # Stwórz nowy thread
        new_thread = openai_client.beta.threads.create()
        
    
        return {
            "status": "success",
            "message": "Chat history cleared",
            "old_thread_id": old_thread_id,
            "new_thread_id": new_thread.id
        }
        
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)