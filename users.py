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


def get_or_create_assistant(firebase_uid: str) -> str:
    
    db = getDb()
    
    user_doc = db.collection("users_assistants").document(firebase_uid).get()
    if user_doc.exists and "assistant_id" in user_doc.to_dict():
        return user_doc.to_dict()["assistant_id"]
    
    
    file = client.files.create(
        file=open(PROMPT_PATH, "rb"), 
        purpose="assistants"
    )
    
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_instructions = f.read()

    assistant = client.beta.assistants.create(
        name=f"ReceiptAnalyzer_{firebase_uid}",
        instructions=system_instructions,
        model="gpt-4o",
        tools=[{"type": "file_search"}],
        tool_resources={
            "file_search": {
                "vector_stores": [
                    {
                        "file_ids": [file.id]
                    }
                ]
            }
        }
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


