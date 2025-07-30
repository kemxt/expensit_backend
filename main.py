import json
from typing import Optional, List
from datetime import datetime
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, MetaData, Table, desc, text, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.sql import select, join
from sqlalchemy.dialects.postgresql import insert
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import sessionmaker
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai_service import router as openai_router
import numpy as np

app = FastAPI()
load_dotenv()
app.include_router(openai_router, prefix="/openai")
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
metadata = MetaData()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Ładowanie istniejących tabel
products = Table("products", metadata, autoload_with=engine)
prices = Table("prices", metadata, autoload_with=engine)

# Nowe tabele dla paragonów i produktów z paragonów
receipts = Table("receipts", metadata,
    Column("id", Integer, primary_key=True),
    Column("store_name", String),
    Column("document_type", String),
    Column("payment_method", String),
    Column("total_amount", Float),
    Column("purchase_date", DateTime),
    Column("created_at", DateTime, default=datetime.utcnow),
    extend_existing=True
)

receipt_products = Table("receipt_products", metadata,
    Column("id", Integer, primary_key=True),
    Column("receipt_id", Integer, ForeignKey("receipts.id")),
    Column("product_id", Integer, ForeignKey("products.id")),  # Dodany ForeignKey
    Column("product_name", String),
    Column("quantity", Integer),
    Column("unit_price", Float),
    Column("total_price", Float),
    Column("created_at", DateTime, default=datetime.utcnow),
    extend_existing=True
)

# Utworzenie tabel jeśli nie istnieją
metadata.create_all(engine)

class ProductIn(BaseModel):
    name: str
    price: float
    description: Optional[str] = None
    embedding: Optional[list[float]] = None

# Modele Pydantic
class ReceiptProductCreate(BaseModel):
    product_id: Optional[int] = None  # Dodane pole
    quantity: int
    unit_price: float
    total_price: float
    purchase_date: Optional[datetime] = None
    product_name: Optional[str] = None

class ReceiptData(BaseModel):
    store_name: str
    document_type: str
    payment_method: Optional[str] = None
    total_amount: float
    purchase_date: datetime
    products: List[ReceiptProductCreate]

class ReceiptProductWithProductIn(BaseModel):
    receipt_id: int
    quantity: int
    unit_price: float
    total_price: float
    purchase_date: datetime
    product: ProductIn

class ReceiptResponse(BaseModel):
    id: int
    store_name: str
    document_type: str
    payment_method: Optional[str]
    total_amount: float
    purchase_date: datetime
    products: List[ReceiptProductCreate]
    

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Istniejące modele
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

class PriceIn(BaseModel):
    id: int
    price: float


def find_similar_product(
    product_name: str,
    product_description: Optional[str] = None,
    query_embedding: Optional[np.ndarray] = None,
    threshold: float = 0.95
):
    try:
        # Przygotuj embedding jeśli nie został podany
        if query_embedding is None:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            text_to_embed = product_description if product_description else product_name
            query_embedding = model.encode(text_to_embed)
       
        query_embedding = np.array(query_embedding).reshape(1, -1)

        with engine.connect() as conn:
            stmt = select(products.c.id, products.c.name, products.c.embedding)
            result = conn.execute(stmt)

            best_match_id = None
            best_similarity = 0.0

            for row in result:
                existing_embedding = row.embedding

                if isinstance(existing_embedding, str):
                    try:
                        existing_embedding = json.loads(existing_embedding)
                    except json.JSONDecodeError:
                        continue

                if existing_embedding is None:
                    continue

                existing_embedding = np.array(existing_embedding)
                if existing_embedding.ndim == 1:
                    existing_embedding = existing_embedding.reshape(1, -1)

                if existing_embedding.shape == query_embedding.shape:
                    similarity = cosine_similarity(query_embedding, existing_embedding)[0][0]

                    if float(similarity) > best_similarity:
                        best_similarity = float(similarity)
                        best_match_id = row.id

        if best_similarity >= threshold:
            return best_match_id, best_similarity
        else:
            return None, best_similarity

    except Exception as e:
        print(f"Error in find_similar_product: {e}")
        return None, 0.0


def create_or_update_product(product_name: str, unit_price: float, product_description: Optional[str] = None):
    """
    Tworzy nowy produkt lub aktualizuje istniejący na podstawie podobieństwa embeddingów
    
    Returns:
        int: product_id (nowego lub istniejącego produktu)
    """
    # Sprawdź czy istnieje podobny produkt
    similar_product_id, similarity = find_similar_product(product_name, product_description)
    
    with engine.connect() as conn:
        if similar_product_id and similarity >= 0.95:
            print(f"Found similar product (ID: {similar_product_id}, similarity: {similarity:.3f}) for '{product_name}'")
            
            # Aktualizuj cenę istniejącego produktu jeśli jest różna
            current_price_result = conn.execute(
                select(prices.c.price).where(prices.c.id == similar_product_id)
            ).first()
            
            if current_price_result and current_price_result.price != unit_price:
                # Aktualizuj cenę
                conn.execute(
                    text("UPDATE prices SET price = :price WHERE id = :id"),
                    {"price": unit_price, "id": similar_product_id}
                )
                print(f"Updated price for product ID {similar_product_id}: {current_price_result.price} -> {unit_price}")
            
            conn.commit()
            return similar_product_id
        else:
            print(f"Creating new product '{product_name}' (best similarity: {similarity:.3f})")
            
            # Utwórz embedding dla nowego produktu
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            text_to_embed = product_description if product_description else product_name
            embedding = model.encode(text_to_embed, batch_size=384).tolist()
            
            # Dodaj nowy produkt
            product_result = conn.execute(
                text("INSERT INTO products (name, embedding, description) VALUES (:name, :embedding, :description) RETURNING id"),
                {
                    "name": product_name, 
                    "embedding": embedding, 
                    "description": product_description
                }
            )
            new_product_id = product_result.scalar()
            
            # Dodaj cenę
            conn.execute(
                text("INSERT INTO prices (id, price) VALUES (:id, :price)"),
                {"id": new_product_id, "price": unit_price}
            )
            
            conn.commit()
            return new_product_id

def create_or_update_product_with_id(product_name: str, unit_price: float, product_description: Optional[str] = None):
    """
    Tworzy nowy produkt lub aktualizuje istniejący, zwraca JSON:
    {
        "id": int,
        "name": str,
        "old_price": float | None,
        "new_price": float
    }
    """
    similar_product_id, similarity = find_similar_product(product_name, product_description)

    with engine.connect() as conn:
        if similar_product_id and similarity >= 0.95:
            # Produkt podobny istnieje – pobieramy jego aktualną cenę
            result = conn.execute(
                select(prices.c.price).where(prices.c.id == similar_product_id)
            ).first()

            current_price = float(result.price) if result else 0.0
            price_changed = current_price != unit_price

            if price_changed:
                conn.execute(
                    text("UPDATE prices SET price = :price WHERE id = :id"),
                    {"price": unit_price, "id": similar_product_id}
                )
                conn.commit()

            return {
                "id": similar_product_id,
                "name": product_name,
                "old_price": current_price if price_changed else None,
                "new_price": unit_price
            }

        else:
            # Tworzymy nowy produkt
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            text_to_embed = product_description if product_description else product_name
            embedding = model.encode(text_to_embed).tolist()

            # Wstawiamy produkt
            product_result = conn.execute(
                text("INSERT INTO products (name, embedding, description) VALUES (:name, :embedding, :description) RETURNING id"),
                {
                    "name": product_name,
                    "embedding": embedding,
                    "description": product_description
                }
            )
            new_product_id = product_result.scalar()

            # Wstawiamy cenę
            conn.execute(
                text("INSERT INTO prices (id, price) VALUES (:id, :price)"),
                {"id": new_product_id, "price": unit_price}
            )
            conn.commit()

            return {
                "id": new_product_id,
                "name": product_name,
                "old_price": None,
                "new_price": unit_price
            }
# Funkcja pomocnicza do parsowania daty
def parse_date_string(date_str: str) -> datetime:
    """Parsuje string daty do obiektu datetime"""
    try:
        # Próbujemy różne formaty
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%d-%m-%Y %H:%M:%S",
            "%d-%m-%Y",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
                
        # Jeśli żaden format nie pasuje, zwróć obecną datę
        return datetime.now()
    except:
        return datetime.now()

# Istniejące endpointy
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}
@app.post("/products/compare")
def check_product(product: ProductIn):
    # Przygotuj embedding
    if product.embedding:
        query_embedding = np.array(product.embedding)
    else:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        text_to_embed = product.description or product.name
        query_embedding = model.encode(text_to_embed)

    # Szukaj podobnego produktu
    similar_product_id, similarity = find_similar_product(product.name, product.description, query_embedding)
    # print(similar_product_id);
    # print(similarity)
    if similar_product_id and similarity >= 0.95:
        with engine.connect() as conn:
            result = conn.execute(
                select(products.c.name, prices.c.price).where(prices.c.id == similar_product_id)
            ).first()

            if result:
                old_price = float(result.price)
                price_changed = old_price != product.price
                return {
                    "exists": True,
                    "id": similar_product_id,
                    "name": result.name,
                    "old_price": old_price,
                    "new_price": product.price,
                    "price_changed": price_changed,
                    "similarity": similarity
                }

    return {
        "exists": False,
        "id": None,
        "name": None,
        "old_price": None,
        "new_price": product.price,
        "price_changed": None,
        "similarity": similarity
    }

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
            select(products.c.id, products.c.name, prices.c.price, products.c.embedding, products.c.description)
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
            embedding = model.encode(product_data.description, batch_size=batch_size)

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

@app.post("/product/add-products-bulk")
def add_products_bulk(products_data: list[ProductIn]):
    """
    Dodaje listę produktów do bazy danych
    """
    results = []
    model = None
    
    with engine.connect() as conn:
        for product_data in products_data:
            try:
                if product_data.embedding is None:
                    if model is None:
                        batch_size = 384
                        model = SentenceTransformer(
                            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"
                        )
                    
                    if product_data.description is None:
                        print(f'embedding from name for product: {product_data.name}')
                        embedding = model.encode(product_data.name, batch_size=384)
                    else:
                        print(f'embedding from description for product: {product_data.name}')
                        embedding = model.encode(product_data.description, batch_size=384)
                    
                    embedding = embedding.tolist()
                else:
                    embedding = product_data.embedding
                
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
                
                results.append({
                    "id": new_id,
                    "name": product_data.name,
                    "status": "inserted"
                })
                
            except Exception as e:
                results.append({
                    "name": product_data.name,
                    "status": "error",
                    "error": str(e)
                })
                print(f"Error inserting product {product_data.name}: {e}")
        
        conn.commit()
    
    return {
        "total_processed": len(products_data),
        "successful_inserts": len([r for r in results if r["status"] == "inserted"]),
        "errors": len([r for r in results if r["status"] == "error"]),
        "results": results
    }

# ZMODYFIKOWANE ENDPOINTY DLA PARAGONÓW

@app.post("/receipt/add", response_model=ReceiptResponse)
def add_receipt(receipt_data: ReceiptData):
    try:
        with engine.connect() as conn:
            purchase_date = receipt_data.purchase_date
            if isinstance(purchase_date, str):
                purchase_date = parse_date_string(purchase_date)

            # Insert paragon
            receipt_result = conn.execute(
                text("""
                    INSERT INTO receipts (store_name, document_type, payment_method, total_amount, purchase_date)
                    VALUES (:store_name, :document_type, :payment_method, :total_amount, :purchase_date)
                    RETURNING id, created_at
                """),
                {
                    "store_name": receipt_data.store_name,
                    "document_type": receipt_data.document_type,
                    "payment_method": receipt_data.payment_method,
                    "total_amount": receipt_data.total_amount,
                    "purchase_date": purchase_date
                }
            )
            receipt_row = receipt_result.first()
            receipt_id = receipt_row.id

            # Przetwórz każdy produkt z paragonu
            response_products = []
            for product in receipt_data.products:
                print("produkt:")
                print(product.product_name)
                product_id = create_or_update_product(
                    product_name=product.product_name,
                    unit_price=product.unit_price,
                    product_description=None  
                )
                
                # Dodaj produkt do receipt_products z product_id
                conn.execute(
                    text("""
                        INSERT INTO receipt_products 
                        (receipt_id, product_id, product_name, quantity, unit_price, total_price)
                        VALUES (:receipt_id, :product_id, :product_name, :quantity, :unit_price, :total_price)
                    """),
                    {
                        "receipt_id": receipt_id,
                        "product_id": product_id,
                        "product_name": product.product_name,
                        "quantity": product.quantity,
                        "unit_price": product.unit_price,
                        "total_price": product.total_price
                    }
                )
                
                response_products.append(
                    ReceiptProductCreate(
                        product_id=product_id,
                        quantity=product.quantity,
                        unit_price=product.unit_price,
                        total_price=product.total_price,
                        purchase_date=purchase_date,
                        product_name=product.product_name
                    )
                )

            conn.commit()

            return ReceiptResponse(
                id=receipt_id,
                store_name=receipt_data.store_name,
                document_type=receipt_data.document_type,
                payment_method=receipt_data.payment_method,
                total_amount=receipt_data.total_amount,
                purchase_date=purchase_date,
                products=response_products,
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding receipt: {str(e)}")

@app.get("/receipts", response_model=List[ReceiptResponse])
def get_receipts(limit: int = 50, offset: int = 0):
    with engine.connect() as conn:
        # Pobieranie paragonów
        receipts_stmt = select(receipts).order_by(desc(receipts.c.created_at)).limit(limit).offset(offset)
        receipts_result = conn.execute(receipts_stmt)
        receipt_list = []
        
        for receipt_row in receipts_result:
            # Pobieranie produktów dla każdego paragonu z JOIN do products
            products_stmt = select(
                receipt_products.c.product_id,
                receipt_products.c.product_name,
                receipt_products.c.quantity,
                receipt_products.c.unit_price,
                receipt_products.c.total_price,
                receipt_products.c.created_at
            ).where(receipt_products.c.receipt_id == receipt_row.id)
            
            products_result = conn.execute(products_stmt)
            products_list = [
                ReceiptProductCreate(
                    product_id=product_row.product_id,
                    product_name=product_row.product_name,
                    quantity=product_row.quantity,
                    unit_price=product_row.unit_price,
                    total_price=product_row.total_price,
                    purchase_date=receipt_row.purchase_date,
                )
                for product_row in products_result
            ]
            
            receipt_list.append(ReceiptResponse(
                id=receipt_row.id,
                store_name=receipt_row.store_name,
                document_type=receipt_row.document_type,
                payment_method=receipt_row.payment_method,
                total_amount=receipt_row.total_amount,
                purchase_date=receipt_row.purchase_date,
                products=products_list,
            ))
        
        return receipt_list

@app.get("/receipt/{receipt_id}", response_model=ReceiptResponse)
def get_receipt(receipt_id: int):
    """
    Pobiera konkretny paragon z produktami
    """
    with engine.connect() as conn:
        # Pobieranie paragonu
        receipt_stmt = select(receipts).where(receipts.c.id == receipt_id)
        receipt_result = conn.execute(receipt_stmt).first()
        
        if not receipt_result:
            raise HTTPException(status_code=404, detail="Receipt not found")
        
        # Pobieranie produktów z receipt_products z informacjami o produkcie
        products_stmt = select(
            receipt_products.c.product_id,
            receipt_products.c.product_name,
            receipt_products.c.quantity,
            receipt_products.c.unit_price,
            receipt_products.c.total_price,
        ).where(receipt_products.c.receipt_id == receipt_id)
        
        products_result = conn.execute(products_stmt)
        
        products_list = [
            ReceiptProductCreate(
                product_id=product_row.product_id,
                product_name=product_row.product_name,
                quantity=product_row.quantity,
                unit_price=product_row.unit_price,
                total_price=product_row.total_price,
                purchase_date=receipt_result.purchase_date,
            )
            for product_row in products_result
        ]
        
        return ReceiptResponse(
            id=receipt_result.id,
            store_name=receipt_result.store_name,
            document_type=receipt_result.document_type,
            payment_method=receipt_result.payment_method,
            total_amount=receipt_result.total_amount,
            purchase_date=receipt_result.purchase_date,
            products=products_list,
        )