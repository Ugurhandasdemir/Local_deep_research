import chromadb
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import os
import base64
import io
import sqlite3
from datetime import datetime
from PyPDF2 import PdfReader
from urllib.parse import quote
from vector_indexes import index_document_for_all_scenarios, search_vector_index

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "DB", "chorame", "yerel_veritabani")
PDF_UPLOAD_DIR = os.path.join(BASE_DIR, "data", "dataset_pdf")
PDF_DB_PATH = os.path.join(BASE_DIR, "pdfs.db")

class GirdiVerisi(BaseModel):
    input: str
    model: str = "ministral-3:3b"  # model identifier sent from client
    scenario: str = "balanced"
    db: str | None = None
    embedding: str | None = None

class PDFMetniVerisi(BaseModel):
    pdf_metni: str
    pdf_adi: str = "Yüklenen PDF"

class PDFBase64Verisi(BaseModel):
    pdf_base64: str
    pdf_adi: str = "Yüklenen PDF"

os.makedirs(PDF_UPLOAD_DIR, exist_ok=True)

COLLECTION_NAME = "dokumanlarim"

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


def safe_pdf_filename(filename: str) -> str:
    safe_name = os.path.basename(filename or "").strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Geçerli bir PDF dosya adı gerekli")
    if not safe_name.lower().endswith(".pdf"):
        safe_name = f"{safe_name}.pdf"
    return safe_name


def ensure_pdf_table() -> None:
    with sqlite3.connect(PDF_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pdfs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                content BLOB,
                uploaded_at TEXT
            )
            """
        )


def save_pdf_to_db(filename: str, content: bytes) -> None:
    ensure_pdf_table()
    with sqlite3.connect(PDF_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO pdfs(filename, content, uploaded_at)
            VALUES (?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                content = excluded.content,
                uploaded_at = excluded.uploaded_at
            """,
            (filename, sqlite3.Binary(content), datetime.now().isoformat(timespec="seconds")),
        )


@app.get("/pdfs/{filename:path}")
def get_pdf(filename: str):
    safe_name = safe_pdf_filename(filename)
    file_path = os.path.join(PDF_UPLOAD_DIR, safe_name)

    if os.path.isfile(file_path):
        return FileResponse(file_path, media_type="application/pdf", filename=safe_name)

    ensure_pdf_table()
    with sqlite3.connect(PDF_DB_PATH) as conn:
        row = conn.execute(
            "SELECT content FROM pdfs WHERE filename = ?",
            (safe_name,),
        ).fetchone()

    if row and row[0]:
        return Response(content=row[0], media_type="application/pdf")

    raise HTTPException(status_code=404, detail="PDF bulunamadı")


def ollama(context_text: str, user_query: str = "", is_deep_research: bool = False, model_name: str = "ministral-3:3b") -> str:
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

    # Qwen gibi büyük modeller varsayılan olarak çok fazla VRAM kullanıyor
    # num_ctx ile context penceresini küçülttük, num_predict ile max cevap uzunluğunu sınırladık
    model = OllamaLLM(
        model=model_name,
        temperature=temp,
        num_ctx=2048,       # varsayılan 32k yerine 2048 token — VRAM çok düşüyor
        num_predict=512,    # cevap en fazla 512 token olsun
    )
    prompt = ChatPromptTemplate.from_template(template_str)
    
    chain = prompt | model
    return chain.invoke(prompt_data)

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)

    # Mevcut koleksiyon Chroma'nın default embedding ayarıyla kaydedilmiş.
    # Burada yeni embedding_function vermek ChromaDB'de config conflict hatası üretir.
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

def hnsw_vector_search(queries: List[str], top_k: int = 5):
    collection = get_collection()

    tum_sonuclar = []

    for q in queries:
        results = collection.query(query_texts=[q], n_results=top_k, include=["documents", "metadatas", "distances"])
        
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


def upsert_default_chroma_chunks(pdf_adi: str, full_text: str) -> int:
    collection = get_collection()

    chunk_size = 3000
    chunks = [
        full_text[i:i + chunk_size]
        for i in range(0, len(full_text), chunk_size)
        if full_text[i:i + chunk_size].strip()
    ]

    if not chunks:
        return 0

    collection.upsert(
        documents=chunks,
        metadatas=[
            {"filename": pdf_adi, "source": pdf_adi, "chunk": idx, "origin": "upload"}
            for idx in range(1, len(chunks) + 1)
        ],
        ids=[f"default:{pdf_adi}:{idx}" for idx in range(1, len(chunks) + 1)],
    )
    return len(chunks)


def pdf_processed_message(pdf_adi: str, index_results: List[Dict[str, Any]]) -> str:
    targets = [r for r in index_results if r.get("scenario") != "all"]
    success_count = sum(1 for r in targets if r.get("status") == "success")
    total_count = len(targets)
    if not total_count:
        return f"{pdf_adi} başarıyla işlendi"
    return f"{pdf_adi} başarıyla işlendi ({success_count}/{total_count} vektör indeks güncellendi)"

@app.post("/ask/question/ai")
def askQuestionAI(veri: GirdiVerisi) -> Dict[str, Any]: 
    try:
        if not veri.input or not veri.input.strip():
            return {
                "status": "error",
                "message": "Soru boş olamaz"
            }
        
        search_results = search_vector_index(
            veri.input,
            scenario=veri.scenario,
            db=veri.db,
            embedding=veri.embedding,
        )
        
        if not search_results:
            print("! ChromaDB'de ilgili doküman bulunamadı", flush=True)
            return {"status": "error", "message": "İlgili doküman bulunamadı. Lütfen PDF yükleyiniz."}


        context_for_ai = ""
        for res in search_results:
            context_for_ai += f"\n--- DOSYA ADI: {res['dosya_adi']} ---\nİÇERİK: {res['metin']}\n"

        final_report = ollama(
            context_text=context_for_ai, 
            user_query=veri.input, 
            is_deep_research=True,
            model_name=veri.model
        )

        sources = []
        for res in search_results:
            filename = res["dosya_adi"] or ""
            sources.append({
                "file": filename,
                "url": f"http://127.0.0.1:8000/pdfs/{quote(filename, safe='')}",
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
        
        pdf_adi = safe_pdf_filename(veri.pdf_adi or "Yüklenen PDF.pdf")
        
        chunks_added = upsert_default_chroma_chunks(pdf_adi, veri.pdf_metni)
        index_results = index_document_for_all_scenarios(pdf_adi, veri.pdf_metni)

        return {
            "status": "success",
            "message": pdf_processed_message(pdf_adi, index_results),
            "document_name": pdf_adi,
            "chunks_added": chunks_added,
            "total_characters": len(veri.pdf_metni),
            "index_results": index_results
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
        
        pdf_adi = safe_pdf_filename(veri.pdf_adi or "Yüklenen PDF.pdf")

        try:
            pdf_bytes = base64.b64decode(veri.pdf_base64)
            print(f"  Dosya boyutu: {len(pdf_bytes)} bytes", flush=True)
        except Exception as decode_error:
            return {
                "status": "error",
                "message": f"Base64 decode hatası: {decode_error}"
            }

        try:
            file_path = os.path.join(PDF_UPLOAD_DIR, pdf_adi)
            with open(file_path, "wb") as buffer:
                buffer.write(pdf_bytes)
            save_pdf_to_db(pdf_adi, pdf_bytes)
        except Exception as save_error:
            return {
                "status": "error",
                "message": f"PDF kaydetme hatası: {save_error}"
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
            chunks_added = upsert_default_chroma_chunks(pdf_adi, full_text)
            index_results = index_document_for_all_scenarios(pdf_adi, full_text)

            return {
                "status": "success",
                "message": pdf_processed_message(pdf_adi, index_results),
                "document_name": pdf_adi,
                "chunks_added": chunks_added,
                "total_characters": len(full_text),
                "pages_processed": page_count,
                "index_results": index_results
            }

        except Exception as index_error:
            return {
                "status": "error",
                "message": f"Vektör DB ekleme hatası: {index_error}"
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
        
        pdf_adi = safe_pdf_filename(file.filename)
        file_path = os.path.join(PDF_UPLOAD_DIR, pdf_adi)
        
        
        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        save_pdf_to_db(pdf_adi, contents)

        try:
            pdf_reader = PdfReader(io.BytesIO(contents))
            page_count = len(pdf_reader.pages)
            full_text = ""
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                full_text += text + f"\n---PAGE {page_num}---\n"

            if not full_text.strip():
                return {
                    "status": "error",
                    "message": "PDF kaydedildi ama metin çıkarılamadı",
                    "filename": pdf_adi
                }

            chunks_added = upsert_default_chroma_chunks(pdf_adi, full_text)
            index_results = index_document_for_all_scenarios(pdf_adi, full_text)
        except Exception as index_error:
            return {
                "status": "error",
                "message": f"PDF kaydedildi ama vektör DB ekleme hatası: {index_error}",
                "filename": pdf_adi
            }

        return {
            "status": "success",
            "message": pdf_processed_message(pdf_adi, index_results),
            "filename": pdf_adi,
            "chunks_added": chunks_added,
            "total_characters": len(full_text),
            "pages_processed": page_count,
            "index_results": index_results
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
        
        
        summary_result = ollama(veri.input, model_name=veri.model)
        
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
