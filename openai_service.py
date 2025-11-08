import asyncio

import os
from pathlib import Path
import time
from fastapi import Body, FastAPI, HTTPException, APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
import json
from dotenv import load_dotenv
from pydantic import BaseModel, HttpUrl
import firebase_admin
from firebase_admin import credentials, storage, auth
from fastapi import Body, FastAPI, HTTPException, Depends, Header
from typing import Optional
from users import get_or_create_assistant, get_or_create_thread
from google.cloud import vision

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "config/expensit-10546-firebase-adminsdk-q25nr-1769d00b3c.json"


router = APIRouter()

load_dotenv()

app = FastAPI()


class EmbeddingRequest(BaseModel):
    text: str

class ImageAnalysisRequest(BaseModel):
    image_url: HttpUrl
    


cred = credentials.Certificate("config/expensit-10546-firebase-adminsdk-q25nr-1769d00b3c.json")
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

def load_receipt_prompt():
    """Wczytuje prompt z pliku tekstowego."""
    prompt_file = Path(__file__).parent / "prompt.txt"
    
    if prompt_file.exists():
        prompt = prompt_file.read_text(encoding='utf-8')
        print(f"✅ Loaded prompt from file: {len(prompt)} characters")
        return prompt
    else:
        print(f"⚠️  Prompt file not found: {prompt_file}")
        
        return os.getenv("GPT_PROMPT", "Extract receipt data to JSON.")

@router.post("/analyze-image")
async def analyze_image(
    image_data: dict = Body(...),
    user_token: dict = Depends(verify_firebase_token)
):
    start_time = time.time()
    
    try:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        user_id = user_token["uid"]
        image_path = image_data["image_path"].strip().lstrip('/')
        if not image_path.startswith(f"users/{user_id}/"):
            return JSONResponse(status_code=403, content={"error": "Access denied"})

        bucket = storage.bucket('expensit-10546.appspot.com')
        blob = bucket.blob(image_path)
        
        if not blob.exists():
            return JSONResponse(status_code=404, content={"error": "Image not found"})
        
       
        print(f"⏱️ Running Vision OCR...")
        ocr_start = time.time()
        vision_client = vision.ImageAnnotatorClient()
        image_bytes = await asyncio.to_thread(blob.download_as_bytes)
        image = vision.Image(content=image_bytes)
        response = await asyncio.to_thread(
            vision_client.document_text_detection, 
            image=image
        )
        
        if not response.full_text_annotation:
            return JSONResponse(status_code=400, content={"error": "No text found"})
        
        raw_text = response.full_text_annotation.text
        
        print(f"✅ OCR done in {time.time() - ocr_start:.2f}s ({len(raw_text)} chars)")
        
        # KROK 2: GPT-4o strukturyzuje tekst (~3-5s, szybsze niż analiza obrazu)
        print(f"⏱️ Structuring with GPT-4o...")
        gpt_start = time.time()
        system_prompt = load_receipt_prompt()
        print(f"⏱️  Structuring with GPT-4o (prompt: {len(system_prompt)} chars)...")

        structure_response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"""Structure this receipt OCR text into JSON. 
Extract ALL items with exact prices (including decimal places).

OCR TEXT:
{raw_text}

Return structured JSON."""
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=1500
        )
        
        print(f"✅ Structured in {time.time() - gpt_start:.2f}s")
        
        receipt_json = json.loads(structure_response.choices[0].message.content)
        
        # Zapis w tle
        receipt_message = f"Receipt: {receipt_json.get('shop', 'Unknown')} - {receipt_json.get('total', 0)} {receipt_json.get('currency', 'PLN')}"
        asyncio.create_task(save_receipt_to_history(user_id, receipt_json, receipt_message))
        
        elapsed = time.time() - start_time
        print(f"🎉 Total time: {elapsed:.2f}s")
        
        return {
            "status": "success",
            "analysis": receipt_json,
            "processing_time": f"{elapsed:.2f}s",
            "method": "vision_ocr"
        }
        
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500, 
            content={"error": str(e), "traceback": traceback.format_exc()}
        )
# Pomocnicza funkcja async do zapisu (nie blokuje odpowiedzi)
async def save_receipt_to_history(user_id: str, receipt_json: dict, message: str):
    """
    Zapisuje paragon do historii konwersacji w tle.
    """
    try:
        # Zapisz do cache/bazy
        save_message_to_db(user_id, "user", f"Receipt uploaded: {json.dumps(receipt_json)}")
        save_message_to_db(user_id, "assistant", message)
        
        print(f"✅ Receipt saved for user {user_id}: {receipt_json.get('shop')}")
    except Exception as e:
        print(f"❌ Error saving receipt: {e}")
        
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
        products = request.get("products", {})
        
        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"error": "Message cannot be empty"}
            )
        
        print(f"User: {user_id}")
        print(f"Message: {user_message}")
        print(f"Products count: {len(products)}")
        
        # Pobierz historię konwersacji z bazy danych
        conversation_history = get_conversation_history(user_id)
        
        # Przygotuj system prompt z kontekstem produktów
        system_message = {
            "role": "system",
            "content": f"""Jesteś pomocnym asystentem finansowym. Pomagasz użytkownikowi zarządzać wydatkami i analizować zakupy.

KONTEKST PRODUKTÓW UŻYTKOWNIKA:
{json.dumps(products, indent=2, ensure_ascii=False)}

Odpowiadaj na pytania użytkownika wykorzystując ten kontekst. 
Gdy użytkownik pyta o swoje zakupy, wydatki lub produkty, odnosi się do danych powyżej.
Bądź konkretny, pomocny i przyjaźnie nastawiony."""
        }
        
        # Zbuduj pełną listę wiadomości
        messages = [system_message] + conversation_history + [
            {"role": "user", "content": user_message}
        ]
        
        # Generator do streamowania odpowiedzi
        async def generate():
            full_response = ''
            try:
                stream = openai_client.chat.completions.create(
                    model="gpt-4o",  # lub "gpt-4o", "gpt-3.5-turbo"
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=1500
                )
                
                for chunk in stream:
                    
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        
                        # Wyślij fragment do frontendu
                        yield f"data: {json.dumps({'type': 'content', 'text': content}, ensure_ascii=False)}\n\n"
                
                # Zapisz wiadomości do bazy danych
                save_message_to_db(user_id, "user", user_message)
                save_message_to_db(user_id, "assistant", full_response)
                
               
                print(f"Full response: {full_response}")
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                print(f"Error in generate: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        
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
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


# Funkcje pomocnicze do zarządzania historią konwersacji

def get_conversation_history(user_id: str, max_messages: int = 20) -> list:
    """
    Pobiera historię konwersacji z bazy danych.
    Ogranicz do ostatnich max_messages wiadomości, żeby nie przekroczyć limitu tokenów.
    """
    # Przykład z Firestore:
    from google.cloud import firestore
    db = firestore.Client()
    
    messages_ref = db.collection('users').document(user_id).collection('messages')
    messages = messages_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(max_messages).stream()
    
    history = []
    for msg in messages:
        data = msg.to_dict()
        history.append({
            "role": data["role"],
            "content": data["content"]
        })
    
   
    return list(reversed(history))


def save_message_to_db(user_id: str, role: str, content: str):
    """
    Zapisuje wiadomość do bazy danych.
    """
    from google.cloud import firestore
    from datetime import datetime
    
    db = firestore.Client()
    
    db.collection('users').document(user_id).collection('messages').add({
        'role': role,
        'content': content,
        'timestamp': datetime.utcnow()
    })

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