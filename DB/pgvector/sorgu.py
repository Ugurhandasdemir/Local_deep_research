import psycopg2
from sentence_transformers import SentenceTransformer
import time

# PostgreSQL bağlantısı
def connect_db():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="vector_db",
            user="postgres",
            password="yeni_sifre",
            port="5432",
            connect_timeout=10
        )
        conn.set_session(autocommit=True)
        return conn
    except Exception as e:
        print(f"✗ Bağlantı hatası: {e}")
        return None

# Embedding modeli
model = SentenceTransformer("all-MiniLM-L6-v2")

# Kayıt sayısını kontrol et
def check_record_count(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM documents;")
        count = cursor.fetchone()[0]
        return count
    except Exception as e:
        print(f"✗ Kayıt sayısı kontrol hatası: {e}")
        return 0
    finally:
        cursor.close()

# Cosine Distance ile arama (<=>)
def search_cosine(conn, query_text, limit=5):
    """Cosine distance kullanarak vektör araması yapar"""
    cursor = conn.cursor()
    query_vector = model.encode(query_text).tolist()
    embedding_str = '[' + ','.join(f'{x:.8f}' for x in query_vector) + ']'
    
    try:
        cursor.execute("""
            SELECT id, chunk_id, doc_id, filename, metin, 
                   1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (embedding_str, embedding_str, limit))
        results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"✗ Cosine arama hatası: {e}")
        return []
    finally:
        cursor.close()

# L2 (Euclidean) Distance ile arama (<->)
def search_l2(conn, query_text, limit=5):
    """L2 (Euclidean) distance kullanarak vektör araması yapar"""
    cursor = conn.cursor()
    query_vector = model.encode(query_text).tolist()
    embedding_str = '[' + ','.join(f'{x:.8f}' for x in query_vector) + ']'
    
    try:
        cursor.execute("""
            SELECT id, chunk_id, doc_id, filename, metin, 
                   embedding <-> %s::vector AS distance
            FROM documents
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
        """, (embedding_str, embedding_str, limit))
        results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"✗ L2 arama hatası: {e}")
        return []
    finally:
        cursor.close()

# Inner Product ile arama (<#>)
def search_inner_product(conn, query_text, limit=5):
    """Inner product kullanarak vektör araması yapar"""
    cursor = conn.cursor()
    query_vector = model.encode(query_text).tolist()
    embedding_str = '[' + ','.join(f'{x:.8f}' for x in query_vector) + ']'
    
    try:
        cursor.execute("""
            SELECT id, chunk_id, doc_id, filename, metin, 
                   (embedding <#> %s::vector) * -1 AS similarity
            FROM documents
            ORDER BY embedding <#> %s::vector
            LIMIT %s;
        """, (embedding_str, embedding_str, limit))
        results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"✗ Inner product arama hatası: {e}")
        return []
    finally:
        cursor.close()

# Filtrelenmiş arama (doc_id ile)
def search_with_filter(conn, query_text, doc_id=None, limit=5):
    """Doc ID filtresi ile cosine similarity araması yapar"""
    cursor = conn.cursor()
    query_vector = model.encode(query_text).tolist()
    embedding_str = '[' + ','.join(f'{x:.8f}' for x in query_vector) + ']'
    
    try:
        if doc_id is not None:
            cursor.execute("""
                SELECT id, chunk_id, doc_id, filename, metin, 
                       1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                WHERE doc_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (embedding_str, doc_id, embedding_str, limit))
        else:
            cursor.execute("""
                SELECT id, chunk_id, doc_id, filename, metin, 
                       1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (embedding_str, embedding_str, limit))
        
        results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"✗ Filtrelenmiş arama hatası: {e}")
        return []
    finally:
        cursor.close()

# İlk N kaydı göster
def show_first_documents(conn, limit=5):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, chunk_id, doc_id, filename, metin
            FROM documents
            ORDER BY id
            LIMIT %s;
        """, (limit,))
        results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"✗ Veri gösterme hatası: {e}")
        return []
    finally:
        cursor.close()

# Index bilgilerini göster
def show_indexes(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT indexname, indexdef 
            FROM pg_indexes 
            WHERE tablename = 'documents';
        """)
        indexes = cursor.fetchall()
        return indexes
    except Exception as e:
        print(f"✗ Index bilgisi hatası: {e}")
        return []
    finally:
        cursor.close()

# Ana fonksiyon
def main():
    conn = connect_db()
    if not conn:
        return
    
    try:
        print("✓ PostgreSQL + pgvector'e bağlanıldı\n")
        
        # Kayıt sayısını kontrol et
        count = check_record_count(conn)
        print(f"✓ Veritabanında {count} adet kayıt bulunmaktadır\n")
        
        if count == 0:
            print("⚠ Veritabanında veri yok! Önce write_vector_database.py çalıştırın.")
            return
        
        # Index bilgilerini göster
        print("="*60)
        print("INDEX BİLGİLERİ")
        print("="*60)
        indexes = show_indexes(conn)
        for idx_name, idx_def in indexes:
            print(f"📋 {idx_name}")
            print(f"   {idx_def[:100]}...")
        print()
        
        # Test sorguları
        queries = [
            "artificial intelligence healthcare",
            "machine learning medical diagnosis",
            "deep learning neural networks"
        ]
        
        for query in queries:
            print("="*60)
            print(f"SORGU: '{query}'")
            print("="*60)
            
            # Cosine Similarity Search
            print("\n📊 COSINE SIMILARITY SEARCH:")
            print("-"*60)
            start = time.time()
            results = search_cosine(conn, query, limit=3)
            search_time = time.time() - start
            print(f"Arama zamanı: {search_time:.4f}s\n")
            
            for idx, (id, chunk_id, doc_id, filename, metin, similarity) in enumerate(results, 1):
                print(f"{idx}. Sonuç (Benzerlik: {similarity:.4f})")
                print(f"   ID: {id} | Doc ID: {doc_id} | Chunk: {chunk_id}")
                print(f"   Filename: {filename}")
                print(f"   Metin: {metin[:150]}...\n")
            
            # L2 Distance Search
            print("📊 L2 (EUCLIDEAN) DISTANCE SEARCH:")
            print("-"*60)
            start = time.time()
            results = search_l2(conn, query, limit=3)
            search_time = time.time() - start
            print(f"Arama zamanı: {search_time:.4f}s\n")
            
            for idx, (id, chunk_id, doc_id, filename, metin, distance) in enumerate(results, 1):
                print(f"{idx}. Sonuç (Distance: {distance:.4f})")
                print(f"   ID: {id} | Doc ID: {doc_id} | Chunk: {chunk_id}")
                print(f"   Filename: {filename}")
                print(f"   Metin: {metin[:150]}...\n")
            
            # Inner Product Search
            print("📊 INNER PRODUCT SEARCH:")
            print("-"*60)
            start = time.time()
            results = search_inner_product(conn, query, limit=3)
            search_time = time.time() - start
            print(f"Arama zamanı: {search_time:.4f}s\n")
            
            for idx, (id, chunk_id, doc_id, filename, metin, similarity) in enumerate(results, 1):
                print(f"{idx}. Sonuç (Similarity: {similarity:.4f})")
                print(f"   ID: {id} | Doc ID: {doc_id} | Chunk: {chunk_id}")
                print(f"   Filename: {filename}")
                print(f"   Metin: {metin[:150]}...\n")
            
            print("\n")
        
        # Belirli bir döküman içinde arama
        print("="*60)
        print("BELİRLİ DÖKÜMANDA ARAMA (Doc ID: 0)")
        print("="*60)
        query = "artificial intelligence"
        start = time.time()
        results = search_with_filter(conn, query, doc_id=0, limit=3)
        search_time = time.time() - start
        print(f"Sorgu: '{query}'")
        print(f"Arama zamanı: {search_time:.4f}s\n")
        
        for idx, (id, chunk_id, doc_id, filename, metin, similarity) in enumerate(results, 1):
            print(f"{idx}. Sonuç (Benzerlik: {similarity:.4f})")
            print(f"   ID: {id} | Doc ID: {doc_id} | Chunk: {chunk_id}")
            print(f"   Filename: {filename}")
            print(f"   Metin: {metin[:150]}...\n")
        
        # İlk 5 kaydı göster
        print("="*60)
        print("İLK 5 KAYIT")
        print("="*60)
        first_records = show_first_documents(conn, limit=5)
        for idx, (id, chunk_id, doc_id, filename, metin) in enumerate(first_records, 1):
            print(f"\n{idx}. ID: {id} | Doc ID: {doc_id} | Chunk: {chunk_id}")
            print(f"   Filename: {filename}")
            print(f"   Metin: {metin[:200]}...")
            
    finally:
        conn.close()
        print("\n✓ PostgreSQL bağlantısı kapatıldı")

if __name__ == "__main__":
    main()