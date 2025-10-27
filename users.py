import datetime
import os
from dotenv import load_dotenv
from firebase_admin import firestore
from openai import OpenAI
from pathlib import Path
load_dotenv()


def getDb():
    db = firestore.client()
    return db 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PROMPT_PATH = Path("prompt.txt")


def get_or_create_assistant(firebase_uid: str,products) -> str:
    
    db = getDb()
    
    user_doc = db.collection("users_assistants").document(firebase_uid).get()
    if user_doc.exists and "assistant_id" in user_doc.to_dict():
        return user_doc.to_dict()["assistant_id"]
    
    
   
    
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_instructions = f.read()

    assistant = client.beta.assistants.create(
        name=f"Financial Assistant for {firebase_uid}",
        instructions=f"""
Jesteś osobistym asystentem finansowym użytkownika. Twoim jedynym zadaniem jest analizowanie historii wydatków, paragonów oraz wspieranie użytkownika w zarządzaniu finansami.

DOSTĘPNE INFORMACJE: 
- Lista wszystkich paragonów użytkownika i produktów w nich zawartych:
{products}

ZASADY:
1. Ignoruj lub uprzejmie odmów odpowiedzi na pytania niezwiązane z tematyką finansów...
2. Nie udzielaj odpowiedzi z zakresu matematyki, historii...
3. Skupiaj się wyłącznie na analizie wydatków...
4. Dzisiejsza data to: {datetime.now()}
5. Przywitaj się z użytkownikiem i zaproponuj mu co może dzisiaj zbadać.
""",
        model="gpt-4o"
    )
    
    
    db.collection("users_assistants").document(firebase_uid).set({
        "assistant_id": assistant.id,
        "created_at": firestore.SERVER_TIMESTAMP
    }, merge=True)
    
    return assistant.id


def get_or_create_thread(user_id: str, agent_id: str) -> str:
    firestore_db = getDb()
    try:
        thread_ref = firestore_db.collection('threads').document(user_id).get()
        
        if thread_ref.exists:
            return thread_ref.to_dict()['thread_id']
        
        thread = client.beta.threads.create()
        
        firestore_db.collection('threads').document(user_id).set({
            'thread_id': thread.id,
            'agent_id': agent_id,
            'created_at': datetime.datetime.now()
        })
        
        return thread.id
        
    except Exception as e:
        print(f"Error creating thread: {str(e)}")
        raise


