from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.sql import select
import os

app = FastAPI()


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://tomek:Haslo123@localhost:6666/porownywarka")

engine = create_engine(DATABASE_URL)
metadata = MetaData(bind=engine)

# Ładowanie tabel
products = Table("products", metadata, autoload_with=engine)
prices = Table("prices", metadata, autoload_with=engine)

# Model odpowiedzi
class ProductResponse(BaseModel):
    id: int
    name: str
    price: float | None

@app.get("/product/{product_id}", response_model=ProductResponse)
def get_product(product_id: int):
    with engine.connect() as conn:
        stmt = (
            select(products.c.id, products.c.name, prices.c.price)
            .select_from(products.join(prices, products.c.id == prices.c.id))
            .where(products.c.id == product_id)
        )
        result = conn.execute(stmt).first()

        if not result:
            raise HTTPException(status_code=404, detail="Product not found")

        return {
            "id": result.id,
            "name": result.name,
            "price": result.price
        }
