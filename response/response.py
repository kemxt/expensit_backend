from pydantic import BaseModel
from typing import List

class ProduktResponse(BaseModel):
    id_produktu: int
    nazwa: str
    cena: float

    class Config:
        orm_mode = True
