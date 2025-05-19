import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, MetaData, Table, desc, text
from sqlalchemy.sql import select
from sqlalchemy.dialects.postgresql import insert
from sentence_transformers import SentenceTransformer
import os

app = FastAPI()


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://tomek:Haslo123@localhost:6666/porownywarka")

engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Ładowanie tabel
products = Table("products", metadata, autoload_with=engine)
prices = Table("prices", metadata, autoload_with=engine)

# Model odpowiedzi
class ProductResponse(BaseModel):
    id: int
    name: str
    price: float | None
    description: str | None
    embedding: list[float]
class SimpleProductResponse(BaseModel):
    id: int
    name: str
    price: float
    description: Optional[str] = Field(None)
    class Config:
        orm_mode = True
        allow_population_by_field_name = True

class ProductsResponseList(BaseModel):
    products: list[SimpleProductResponse]
@app.get("/products", response_model=list[SimpleProductResponse])
def get_products():
    with engine.connect() as conn:
        stmt = select(
            products.c.id,
            products.c.name,
            products.c.description,
            prices.c.price
        ).select_from(
            products.join(prices, products.c.id == prices.c.id)
        )
        result = conn.execute(stmt)
        
        product_list = []
        for row in result:
            
            product_data = {
                "id": row.id,
                "name": row.name,
                "description": getattr(row, 'description', None),
                "price": row.price
            }
            if hasattr(row, 'description') and row.description is not None:
                product_data["description"] = row.description
                
            product_list.append(product_data)
            
        return product_list
@app.get("/product/{product_id}", response_model=ProductResponse)
def get_product(product_id: int):
    with engine.connect() as conn:
        stmt = (
            select(products.c.id, products.c.name, prices.c.price,products.c.embedding,products.c.description)
            .select_from(products.join(prices, products.c.id == prices.c.id))
            .where(products.c.id == product_id)
        )
        result = conn.execute(stmt).first()
        embedding = result.embedding
        if isinstance(embedding, str):
            try:
                embedding = json.loads(embedding)
            except json.JSONDecodeError:
               
                embedding = []
        if not result:
            raise HTTPException(status_code=404, detail="Product not found")

        return {
            "id": result.id,
            "name": result.name,
            "price": result.price,
            "description": result.description,
            "embedding": embedding
        }
class PriceIn(BaseModel):
    id: int
    price: float

@app.post("/product/add-price")
def add_or_update_price(price_data: PriceIn):
    stmt = insert(prices).values(id=price_data.id, price=price_data.price)
    stmt = stmt.on_conflict_do_update(
        index_elements=[prices.c.id],
        set_={"price": price_data.price}
    )

    with engine.connect() as conn:
        conn.execute(stmt)
        conn.commit()

    return {
        "id": price_data.id,
        "price": price_data.price,
        "status": "updated or inserted"
    }
class ProductIn(BaseModel):
    name: str
    price: float
    description: Optional[str] = None
    embedding: Optional[list[float]] = None
@app.post("/product/add-product")
def add_or_update_product(product_data: ProductIn):
    if product_data.embedding is None:
        batch_size = 384
        model = SentenceTransformer(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"
        )
        if (product_data.description is None):
            print('embedding from name')
            embedding = model.encode(product_data.name, batch_size=batch_size)
        else:
            print('embedding from description')
            embedding = embedding = model.encode(product_data.description, batch_size=batch_size)

        embedding = embedding.tolist()
    else:
        embedding = product_data.embedding

    with engine.connect() as conn:
        product_result = conn.execute(
            text("INSERT INTO products (name, embedding, description) VALUES (:name, :embedding, :description) RETURNING id"),
            {"name": product_data.name, "embedding": embedding, "description": product_data.description}
        )
        new_id = product_result.scalar()
        conn.execute(
                text("""
                    INSERT INTO prices (id, price)
                    VALUES (:id, :price)
                """),
                {"id": new_id, "price": product_data.price}
            )
        conn.commit()

    return {
        "id": new_id,
        "name": product_data.name,
        "status": "inserted"
    }