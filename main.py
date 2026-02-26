import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import base64
import io
from PyPDF2 import PdfReader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = r"C:\Users\hakan\Desktop\bitirme\Local_deep_research\DB\chorame\yerel_veritabani"
PDF_UPLOAD_DIR = r"C:\Users\hakan\Desktop\bitirme\Local_deep_research\CUSTOM_DATASET\pdfs"

class GirdiVerisi(BaseModel):
    input: str

class PDFMetniVerisi(BaseModel):
    pdf_metni: str
    pdf_adi: str = "Yüklenen PDF"

class PDFBase64Verisi(BaseModel):
    pdf_base64: str
    pdf_adi: str = "Yüklenen PDF"

os.makedirs(PDF_UPLOAD_DIR, exist_ok=True)

app.mount("/pdfs", StaticFiles(directory=PDF_UPLOAD_DIR), name="pdfs")

COLLECTION_NAME = "dokumanlarim"

MODEL_NAME = "all-MiniLM-L6-v2"
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


def ollama(context_text: str, user_query: str = "", is_deep_research: bool = False) -> str:
    if is_deep_research:
        template_str = """
        System:
        Sen profesyonel bir rapor hazırlayıcısısın. 
        Aşağıdaki kaynak metinleri kullanarak, kullanıcının sorusuna cevap veren akıcı bir Türkçe rapor yaz.

        Kurallar:
        - Asla aynı cümleleri tekrar etme.
        - Eğer metinlerde cevap yoksa "Bilgi bulunamadı" de.
        - Kaynakları metin içinde belirt.

        Kaynaklar: {content}
        Soru: {query}
        """
        prompt_data = {"content": context_text, "query": user_query}
    else:
        template_str = "{content}"
        prompt_data = {"content": context_text}

    temp = 0 if is_deep_research else 0.7
    model = OllamaLLM(model="llama3.2:3b", temperature=temp)
    prompt = ChatPromptTemplate.from_template(template_str)
    
    chain = prompt | model
    return chain.invoke(prompt_data)

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

def hnsw_vector_search(queries: List[str], top_k: int = 5):
    model = SentenceTransformer(MODEL_NAME)
    collection = get_collection()
    
    tum_sonuclar = [] 

    for q in queries:
        q_vec = model.encode(q).tolist()
        results = collection.query(query_embeddings=[q_vec], n_results=top_k, include=["documents", "metadatas", "distances"])
        
        if results['documents']:
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ), 1):
                sim = 1 - dist
                print(f"  {i}. Benzerlik: {sim:.4f} | Dosya: {meta.get('filename')}", flush=True)
                
                tum_sonuclar.append({
                    "sirasi": i,
                    "dosya_adi": meta.get('filename'),
                    "benzerlik": float(f"{sim:.4f}"),
                    "metin": doc
                })

    return tum_sonuclar

@app.post("/ask/question/ai")
def askQuestionAI(veri: GirdiVerisi) -> Dict[str, Any]: 
    try:
        if not veri.input or not veri.input.strip():
            return {
                "status": "error",
                "message": "Soru boş olamaz"
            }
        
        search_results = hnsw_vector_search([veri.input])
        
        if not search_results:
            print("! ChromaDB'de ilgili doküman bulunamadı", flush=True)
            return {"status": "error", "message": "İlgili doküman bulunamadı. Lütfen PDF yükleyiniz."}


        context_for_ai = ""
        for res in search_results:
            context_for_ai += f"\n--- DOSYA ADI: {res['dosya_adi']} ---\nİÇERİK: {res['metin']}\n"

        final_report = ollama(
            context_text=context_for_ai, 
            user_query=veri.input, 
            is_deep_research=True
        )

        sources = []
        for res in search_results:
            sources.append({
                "file": res["dosya_adi"],
                "url": f"http://127.0.0.1:8000/pdfs/{res['dosya_adi']}",
                "score": res["benzerlik"],
                "metin": res["metin"][:200]
            })

        return {
            "status": "success",
            "aiResponse": final_report,
            "sources": sources
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/ingest/pdf")
async def ingestPdf(veri: PDFMetniVerisi) -> Dict[str, Any]:
    try:
        if not veri.pdf_metni or not veri.pdf_metni.strip():
            return {
                "status": "error",
                "message": "PDF metni boş olamaz"
            }
        
        pdf_adi = veri.pdf_adi or "Yüklenen PDF"
        
        collection = get_collection()
        
        chunk_size = 3000
        chunks = [veri.pdf_metni[i:i+chunk_size] 
                 for i in range(0, len(veri.pdf_metni), chunk_size)]
        
        for idx, chunk in enumerate(chunks, 1):
            collection.add(
                documents=[chunk],
                metadatas=[{"filename": pdf_adi, "chunk": idx}],
                ids=[f"{pdf_adi}_{idx}"]
            )
        
        return {
            "status": "success",
            "message": f"{pdf_adi} başarıyla işlendi",
            "document_name": pdf_adi,
            "chunks_added": len(chunks),
            "total_characters": len(veri.pdf_metni)
        }
        
    except Exception as e:
        import traceback
        print(f"! PDF ingest hatası: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/upload/pdf/base64")
async def uploadPdfBase64(veri: PDFBase64Verisi) -> Dict[str, Any]:

    try:
        if not veri.pdf_base64 or not veri.pdf_base64.strip():
            return {
                "status": "error",
                "message": "PDF base64 boş olamaz"
            }
        
        pdf_adi = veri.pdf_adi or "Yüklenen PDF"
        
        try:
            pdf_bytes = base64.b64decode(veri.pdf_base64)
            print(f"  Dosya boyutu: {len(pdf_bytes)} bytes", flush=True)
        except Exception as decode_error:
            return {
                "status": "error",
                "message": f"Base64 decode hatası: {decode_error}"
            }
        
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PdfReader(pdf_file)
            page_count = len(pdf_reader.pages)
            
            
            full_text = ""
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    text = page.extract_text()
                    full_text += text + f"\n---PAGE {page_num}---\n"
                    print(f"  Sayfa {page_num} çıkartıldı", flush=True)
                except Exception as page_error:
                    print(f"  ! Sayfa {page_num} çıkartılamadı: {page_error}", flush=True)
            
            if not full_text.strip():
                return {
                    "status": "error",
                    "message": "PDF'den metin çıkarılamadı"
                }
            
            
        except Exception as extract_error:
            return {
                "status": "error",
                "message": f"PDF metin çıkartma hatası: {extract_error}"
            }
        
        try:
            collection = get_collection()
            
            chunk_size = 3000
            chunks = [full_text[i:i+chunk_size] 
                     for i in range(0, len(full_text), chunk_size)]
            
            for idx, chunk in enumerate(chunks, 1):
                collection.add(
                    documents=[chunk],
                    metadatas=[{"filename": pdf_adi, "chunk": idx}],
                    ids=[f"{pdf_adi}_{idx}"]
                )
            
            
            return {
                "status": "success",
                "message": f"{pdf_adi} başarıyla işlendi",
                "document_name": pdf_adi,
                "chunks_added": len(chunks),
                "total_characters": len(full_text),
                "pages_processed": page_count
            }
            
        except Exception as chroma_error:
            return {
                "status": "error",
                "message": f"ChromaDB ekleme hatası: {chroma_error}"
            }
        
    except Exception as e:
        import traceback
        print(f"! PDF base64 upload hatası: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/upload/pdf")
async def uploadPdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        if not file or not file.filename:
            return {"status": "error", "message": "Dosya seçilmedi"}
            
        if not file.filename.lower().endswith('.pdf'):
            return {
                "status": "error",
                "message": "Sadece PDF dosyaları yüklenebilir"
            }
        
        file_path = os.path.join(PDF_UPLOAD_DIR, file.filename)
        
        
        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        
        return {
            "status": "success",
            "message": f"{file.filename} başarıyla yüklendi",
            "filename": file.filename
        }
    except Exception as e:
        import traceback
        print(f"! PDF upload hatası: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/normal/chat")
async def normalChat(veri: GirdiVerisi) -> Dict[str, Any]:
    try:
        if not veri.input or not veri.input.strip():
            return {
                "status": "error",
                "message": "input boş olamaz"
            }
        
        
        summary_result = ollama(veri.input)
        
        return {
            "status": "success",
            "summary": summary_result
        }
        
    except Exception as e:
        import traceback
        print(f"Hata: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        return {
            "status": "error",
            "message": str(e)
        }
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)