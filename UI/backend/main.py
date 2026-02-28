from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ChromaDB bağlantısı
client = chromadb.PersistentClient(path="../../yerel_veritabani")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_or_create_collection(
    name="dokumanlarim",
    embedding_function=ef
)

class GirdiVerisi(BaseModel):
    girilen_metin: str
    model: str = "ministral-3:3b"

def sorgu_yap(sorgu: str, n_results: int = 5) -> str:
    """ChromaDB'de sorgu yaparak en yakın sonuçları döndürür"""
    arama_sonuclari = collection.query(
        query_texts=[sorgu],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    if not arama_sonuclari['documents'][0]:
        return "Sonuç bulunamadı."
    
    sonuc_metni = f"'{sorgu}' için bulunan sonuçlar:\n\n"
    for i, (doc, meta, dist) in enumerate(zip(
        arama_sonuclari['documents'][0],
        arama_sonuclari['metadatas'][0],
        arama_sonuclari['distances'][0]
    ), 1):
        benzerlik = 1 - dist
        sonuc_metni += f"{i}. Sonuç (Benzerlik: {benzerlik:.2%}):\n"
        sonuc_metni += f"   {doc[:200]}...\n\n"
    
    return sonuc_metni

@app.post("/ask/question/ai")
def askQuestionAI(veri: GirdiVerisi) -> Dict[str, Any]: 
    # ChromaDB'de sorgu yap
    aiResponse = sorgu_yap(veri.girilen_metin)
    
    return {"aiResponse": aiResponse}