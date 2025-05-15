from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel

from response.response import ProduktResponse
from openai_service import router as openai_router
from qdrant_embedings import qdrant_router
DATABASE_URL = "postgresql://postgres:1234@localhost:6666/porownywarka"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

class Produkt(Base):
    __tablename__ = "produkty"
    
    id_produktu = Column(Integer, primary_key=True, index=True)
    nazwa = Column(String, index=True)
    
    # Relacja z tabelą Cena
    ceny = relationship("Cena", back_populates="produkt")
    
class Cena(Base):
    __tablename__ = "ceny"
    
    # Produkt_id jako klucz obcy
    produkt_id = Column(Integer, ForeignKey("produkty.id_produktu"), primary_key=True, index=True)
    
    cena = Column(Float)
    
    # Relacja z tabelą Produkt
    produkt = relationship("Produkt", back_populates="ceny")

# Tworzenie tabeli w bazie
Base.metadata.create_all(bind=engine)


app = FastAPI()
app.include_router(
    openai_router,
    prefix="/openai"
)
app.include_router(qdrant_router)
# Model Pydantic do walidacji
class ProduktSchema(BaseModel):
    nazwa: str
    cena: float

# Dependency do pobierania sesji bazy danych
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 📌 Endpointy API

# Pobierz wszystkie produkty
@app.get("/produkty")
def get_produkty(db: Session = Depends(get_db)):
    produkty = db.query(Produkt).all()  
    
    # Prosta lista słowników z produktami i ich cenami
    response = []
    for produkt in produkty:
        cena = produkt.ceny[0].cena if produkt.ceny else None  # Pobierz cenę (jeśli istnieje)
        
        response.append({
            "id_produktu": produkt.id_produktu,
            "nazwa": produkt.nazwa,
            "cena": cena
        })
    
    return response
@app.get("/ceny")
def get_produkty(db: Session = Depends(get_db)):
    
    produkty = db.query(Cena).all()
    return produkty

@app.get("/produkty/{produkt_id}", response_model=ProduktResponse)
def get_produkt(produkt_id: int, db: Session = Depends(get_db)):
    produkt = db.query(Produkt).filter(Produkt.id_produktu == produkt_id).first()
    if produkt is None:
        raise HTTPException(status_code=404, detail="Produkt nie znaleziony")
    
    cena_row = db.query(Cena.cena).filter(Cena.produkt_id == produkt_id).first()
    
    if cena_row is None:
        raise HTTPException(status_code=404, detail="Cena nie znaleziona")
    
    cena = cena_row[0]  # 🔹 Wyciągamy liczbę float z krotki

    return ProduktResponse(
        id_produktu=produkt.id_produktu,
        nazwa=produkt.nazwa,
        cena=cena
    )
#po nazwie
@app.get("/produkty/nazwa/{nazwa_produktu}", response_model=ProduktResponse)
def get_produkt(nazwa_produktu: str, db: Session = Depends(get_db)):
    produkt = db.query(Produkt).filter(Produkt.nazwa == nazwa_produktu).first()
    if produkt is None:
        raise HTTPException(status_code=404, detail="Produkt nie znaleziony")
    
    cena_row = db.query(Cena.cena).filter(Cena.produkt_id == produkt.id_produktu).first()
    
    if cena_row is None:
        raise HTTPException(status_code=404, detail="Cena nie znaleziona")
    
    cena = cena_row[0]  # 🔹 Wyciągamy liczbę float z krotki

    return ProduktResponse(
        id_produktu=produkt.id_produktu,
        nazwa=produkt.nazwa,
        cena=cena
    )

# Dodaj nowy produkt
@app.post("/produkty")
def create_produkt(produkt: ProduktSchema, db: Session = Depends(get_db)):
    # Tworzenie nowego produktu
    nowy_produkt = Produkt(nazwa=produkt.nazwa)
    
    # Tworzenie nowej ceny i przypisanie jej do produktu
    nowa_cena = Cena(cena=produkt.cena, produkt=nowy_produkt)  # relacja z produktem
    
    # Dodanie produktu i ceny do sesji bazy danych
    db.add(nowy_produkt)
    db.add(nowa_cena)
    
    # Zatwierdzenie zmian
    db.commit()
    
    # Odświeżenie obiektów, aby uzyskać zaktualizowane dane
    db.refresh(nowy_produkt)
    db.refresh(nowa_cena)
    
    return nowy_produkt

@app.delete("/produkty/{produkt_id}")
def delete_produkt(produkt_id: int, db: Session = Depends(get_db)):
    produkt = db.query(Produkt).filter(Produkt.id == produkt_id).first()
    if produkt is None:
        raise HTTPException(status_code=404, detail="Produkt nie znaleziony")
    db.delete(produkt)
    db.commit()
    return {"message": "Produkt usunięty"}
