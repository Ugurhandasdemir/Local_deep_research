import psycopg2
from sentence_transformers import SentenceTransformer
import json
import os
import time

# PostgreSQL bağlantısı
def connect_db():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="vector_db",
            user="postgres",
            password="yeni_sifre",
            port="5432"
        )
        return conn
    except Exception as e:
        print(f"Bağlantı hatası: {e}")
        return None

# Veritabanı ve tablo oluştur
def create_table(conn):
    cursor = conn.cursor()
    try:
        # pgvector extension'ını etkinleştir
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Eski tabloyu sil (temiz başlangıç için)
        cursor.execute("DROP TABLE IF EXISTS documents;")
        
        # Tablo oluştur
        cursor.execute("""
            CREATE TABLE documents (
                id BIGSERIAL PRIMARY KEY,
                chunk_id INT,
                doc_id INT,
                filename VARCHAR(255),
                filepath TEXT,
                metin TEXT,
                embedding vector(384)
            );
        """)
        
        # HNSW index oluştur (cosine distance için)
        cursor.execute("""
            CREATE INDEX ON documents 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)
        
        conn.commit()
        print("✓ Tablo ve index başarıyla oluşturuldu")
    except Exception as e:
        print(f"✗ Tablo oluşturma hatası: {e}")
        conn.rollback()
    finally:
        cursor.close()

# Metni parçalara böl
def metni_parcala(metin, chunk_size=500, overlap=100):
    """Metni çakışmalı parçalara böl"""
    parcalar = []
    kelimeler = metin.split()
    
    for i in range(0, len(kelimeler), chunk_size - overlap):
        parca = ' '.join(kelimeler[i:i + chunk_size])
        if len(parca.strip()) > 50:
            parcalar.append(parca.strip())
    
    return parcalar

# Veri ekle
def insert_documents(conn, veriler):
    cursor = conn.cursor()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    batch_size = 100
    total_inserted = 0
    
    for i in range(0, len(veriler), batch_size):
        batch_veriler = veriler[i:i+batch_size]
        batch_metinler = [v["metin"] for v in batch_veriler]
        
        # Embedding oluştur
        embeddings = model.encode(batch_metinler)
        
        try:
            for j, (veri, embedding) in enumerate(zip(batch_veriler, embeddings)):
                embedding_str = '[' + ','.join(f'{x:.8f}' for x in embedding.tolist()) + ']'
                
                cursor.execute("""
                    INSERT INTO documents (chunk_id, doc_id, filename, filepath, metin, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                """, (
                    veri["chunk_id"], 
                    veri["doc_id"], 
                    veri["filename"], 
                    veri["filepath"], 
                    veri["metin"], 
                    embedding_str
                ))
            
            conn.commit()
            total_inserted += len(batch_veriler)
            print(f"✓ {total_inserted}/{len(veriler)} kayıt eklendi...")
        except Exception as e:
            print(f"✗ Veri ekleme hatası: {e}")
            conn.rollback()
    
    cursor.close()
    print(f"\n✓ Toplam {total_inserted} kayıt başarıyla eklendi")

# Ana fonksiyon
def main():
    # JSON dosyasını oku
    json_dosya_yolu = '/home/ugo/Documents/Python/bitirememe projesi/metin_dosyasi.json'
    
    if not os.path.exists(json_dosya_yolu):
        print(f"✗ HATA: {json_dosya_yolu} dosyası bulunamadı!")
        return
    
    with open(json_dosya_yolu, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Dökümanları çıkar
    documents = data.get("documents", [])
    print(f"✓ Toplam {len(documents)} adet döküman bulundu")
    
    # Tüm parçaları topla
    tum_veriler = []
    
    for doc in documents:
        doc_id = doc.get("id", 0)
        filename = doc.get("filename", "")
        filepath = doc.get("filepath", "")
        full_text = doc.get("full_text", "")
        
        if not full_text or len(full_text.strip()) < 50:
            print(f"⚠ Atlanan döküman (boş veya çok kısa): {filename}")
            continue
        
        # Metni parçala
        parcalar = metni_parcala(full_text, chunk_size=300, overlap=50)
        
        for chunk_idx, parca in enumerate(parcalar):
            tum_veriler.append({
                "metin": parca,
                "chunk_id": chunk_idx,
                "doc_id": doc_id,
                "filename": filename,
                "filepath": filepath
            })
    
    print(f"✓ Toplam {len(tum_veriler)} adet metin parçası oluşturuldu\n")
    
    # Veritabanına bağlan
    conn = connect_db()
    if not conn:
        return
    
    try:
        # Tablo oluştur
        create_table(conn)
        
        # Veri ekle
        print("📝 Veri ekleniyor...\n")
        start_time = time.time()
        insert_documents(conn, tum_veriler)
        insert_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✓ {len(tum_veriler)} adet veri başarıyla pgvector'e kaydedildi!")
        print(f"✓ Toplam ekleme zamanı: {insert_time:.2f}s")
        print(f"{'='*60}")
        
        # Kayıt sayısını kontrol et
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents;")
        count = cursor.fetchone()[0]
        cursor.close()
        
        print(f"\n📊 Tablo Bilgisi:")
        print(f"   Tablo adı: documents")
        print(f"   Toplam kayıt: {count}")
        print(f"✓ İşlem tamamlandı")
            
    finally:
        conn.close()
        print("✓ Veritabanı bağlantısı kapatıldı")

if __name__ == "__main__":
    main()
