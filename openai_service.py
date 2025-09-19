import base64
import datetime
from io import BytesIO
import io
import os
import tempfile
from PIL import Image
from fastapi import Body, FastAPI, HTTPException, APIRouter, requests
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import requests
import json
from dotenv import load_dotenv
from typing import Dict, Any
from pydantic import BaseModel, HttpUrl
import firebase_admin
from firebase_admin import credentials, storage, auth
from fastapi import Body, FastAPI, HTTPException, Depends, Header
from typing import Dict, Optional

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
async def analyze_image(
    image_data: Dict[str, str] = Body(...),
    user_token: dict = Depends(verify_firebase_token)
):
    try:
        user_id = user_token["uid"]
        image_path = image_data["image_path"].strip().lstrip('/')

        if not image_path.lower().endswith(('.jpg', '.jpeg')):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Only JPG/JPEG images are supported"}
            )

        if not image_path.startswith(f"users/{user_id}/"):
            return JSONResponse(
                status_code=403,
                content={"status": "error", "message": "Access denied"}
            )

        bucket = storage.bucket('expensit-10546.appspot.com')
        blob = bucket.blob(image_path)

        if not blob.exists():
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Image not found"}
            )
        
        image_bytes = blob.download_as_bytes()
        
        # Weryfikacja poprawności formatu
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.format.lower() not in ('jpeg', 'jpg'):
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "Image format not JPEG/JPG"}
                )
        except Exception as img_error:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Invalid image: {str(img_error)}"}
            )

        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        if len(base64_image) > 20 * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Image too large (max 20MB)"}
            )

        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json; charset=utf-8"
        }
        
        models_response = requests.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            timeout=10
        )

        
        # Sprawdź czy response jest prawidłowy JSON

        # Dopiero teraz sprawdzaj gpt-4o
        if models_response.status_code == 200:
            data = models_response.json().get('data', [])
            if "gpt-4o" not in [m['id'] for m in data]:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "GPT-4o not available with this API key"}
                )
        else:
            return JSONResponse(
                status_code=models_response.status_code,
                content={"status": "error", "message": f"API error: {models_response.text}"}
            )
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                    "type": "text",
                   "text": os.getenv('GPT_PROMPT') + "\n\nIMPORTANT: Return ONLY valid JSON without markdown formatting or ```json blocks."
                },
                        {
                            "type": "image_url",
                            "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                        }
                    ]
                }
            ]
        }
        print("GPT PROMPT")
        print(os.getenv('GPT_PROMPT'))
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        response.raise_for_status()
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "analysis": response.json()['choices'][0]['message']['content']
            }
        )

    except requests.exceptions.HTTPError as http_err:
        content_types = [
            str(item["type"]) for item in payload["messages"][0]["content"]
        ]
        return JSONResponse(
            status_code=http_err.response.status_code,
            content={
                "status": "error",
                "message": "OpenAI API request failed",
                "details": http_err.response.json(),
                "debug": {
                    "image_size_kb": len(image_bytes) / 1024,
                    "base64_length": len(base64_image),
                    "model_verified": True,
                    "content_types": content_types
                }
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Server error: {str(e)}"}
        )

@router.post("/get-embedding")
async def get_embedding(request: EmbeddingRequest = Body(...)):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"input": request.text, "model": "text-embedding-ada-002"}
            )
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

@router.post("/stream-analyze-image")
async def stream_analyze_image_with_openai(image_request: ImageAnalysisRequest):
    """Stream the analysis of an image using OpenAI's GPT-4o model."""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        prompt = os.getenv('GPT_PROMPT')
        
        if not api_key or not prompt:
            raise HTTPException(status_code=500, detail="API configuration missing")
        
        image_url = str(image_request.image_url)
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ],
            "stream": True
        }
        
        async def generate():
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        yield f"data: {json.dumps({'error': f'OpenAI API error: {response.text}'})}\n\n"
                        return
                    
                    async for chunk in response.aiter_lines():
                        if chunk.startswith('data: '):
                            json_str = chunk[6:].strip()
                            if json_str == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(json_str)
                                content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                yield f"data: {json.dumps({'content': content})}\n\n"
                            except json.JSONDecodeError:
                                continue
        
        return StreamingResponse(generate(), media_type="text/event-stream")
        
    except Exception as e:
        async def generate_error():
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return StreamingResponse(generate_error(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)