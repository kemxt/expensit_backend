from ast import MatchValue
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, FieldCondition, Filter
from sentence_transformers import SentenceTransformer
from typing import List, Optional

# Inicjalizacja routera
qdrant_router = APIRouter(prefix="/qdrant", tags=["Qdrant"])


class Product(BaseModel):
    id: int
    name: str
    price: float
    description: Optional[str]  = None
    category: Optional[str] = None 
    subcategory: Optional[str] = None
    
class SearchQuery(BaseModel):
    text: str
    limit: int = 3

model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(host="localhost", port=6333)
collection_name = "products"

@qdrant_router.on_event("startup")
async def startup():
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384, 
                distance=Distance.COSINE
            )
        )
        # Dodaj przykładowe dane
        sample_products = [
            Product(id=1, name="iPhone 13 Pro", price=4299.99, 
                   description="Flagowy smartfon Apple z potrójnym aparatem"),
            Product(id=2, name="Samsung Galaxy S22", price=3899.00, 
                   description="Telefon Android z doskonałym ekranem AMOLED")
        ]
        await add_products(sample_products)
    except Exception as e:
        print(f"Error during startup: {str(e)}")
@qdrant_router.post("/addProduct/")
async def add_products(products: List[Product]):
    try:
        # Przygotuj teksty do embeddingu - używaj description lub name
        texts_to_embed = [
            p.description if p.description else p.name 
            for p in products
        ]
        
        # Generuj wektory dla wszystkich produktów
        vectors = model.encode(texts_to_embed)
        
        points = [
            PointStruct(
                id=product.id,
                vector=vectors[i].tolist(),
                payload={
                    "name": product.name,
                    "price": product.price,
                    "description": product.description,
                    "category": product.category,
                    "subcategory": product.subcategory,
                    "vector_source": "description" if product.description else "name"  # dla śledzenia źródła
                }
            )
            for i, product in enumerate(products)
        ]
        
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )
        
        return {
            "message": f"Dodano {len(products)} produktów",
            "with_description": sum(1 for p in products if p.description),
            "with_name_only": sum(1 for p in products if not p.description)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@qdrant_router.post("/search/")
async def search_products(query: SearchQuery):
    try:
        query_vector = model.encode(query.text).tolist()
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=query.limit
        )
        return [
            {
                "id": hit.id,
                "name": hit.payload["name"],
                "price": hit.payload["price"],
                "description": hit.payload["description"],
                "score": float(hit.score)
            }
            for hit in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@qdrant_router.post("/search/{category}")
async def search_in_category(
    category: str,
    query: SearchQuery,
    subcategory: Optional[str] = None
):
    try:
        query_vector = model.encode(query.text).tolist()
        
        # Budujemy filtr
        filters = [FieldCondition(key="category", match=MatchValue(value=category))]
        
        if subcategory:
            filters.append(FieldCondition(key="subcategory", match=MatchValue(value=subcategory)))
        
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=Filter(must=filters),
            limit=query.limit
        )

        return [
            {
                "id": hit.id,
                "name": hit.payload["name"],
                "price": hit.payload["price"],
                "description": hit.payload["description"],
                "score": float(hit.score),
                "category": hit.payload.get("category"),
                "subcategory": hit.payload.get("subcategory")
            }
            for hit in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@qdrant_router.get("/products")
async def get_all_products():
    try:
        all_products = client.scroll(
            collection_name='products',
            limit=10000,
            with_payload=True,
            with_vectors=True
        )
        return {
            "status": "success",
            "data": [hit.payload for hit in all_products[0]],
            "count": len(all_products[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    