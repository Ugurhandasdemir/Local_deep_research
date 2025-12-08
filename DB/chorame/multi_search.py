import chromadb
from chromadb.utils import embedding_functions
import os
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import hnswlib
import annoy

class ChromaDBWithMultipleIndexes:
    """ChromaDB + Çoklu Arama Algoritmaları"""
    
    def __init__(self, chroma_path="./hnsw_chroma_db", 
                 collection_name="dokumanlarim",
                 max_elements=50000):
        """
        ChromaDB ve çoklu arama indexleri başlat
        
        Args:
            chroma_path: ChromaDB dosya yolu
            collection_name: ChromaDB koleksiyon adı
            max_elements: Max eleman sayısı
        """
        # ChromaDB'yi başlat
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef
        )
        
        # Model ve embedding boyutu
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_dim = self.model.get_sentence_embedding_dimension()
        self.embedding_dim = embedding_dim
        
        # HNSW
        self.hnsw_index = hnswlib.Index(space='cosine', dim=embedding_dim)
        self.hnsw_index.init_index(
            max_elements=max_elements,
            ef_construction=200,
            M=16
        )
        
        # Annoy (L2 metriği kullanır)
        self.annoy_index = annoy.AnnoyIndex(embedding_dim, metric='angular')
        
        # Flat (Numpy array olarak depolayacağız)
        self.flat_embeddings = []
        self.flat_ids = []
        
        # Tüm indexler için metadata
        self.documents_metadata = {}
        self.counter = 0
    
    def metni_parcala(self, metin, chunk_size=500, overlap=100):
        """Metni çakışmalı parçalara böl"""
        parcalar = []
        kelimeler = metin.split()
        
        for i in range(0, len(kelimeler), chunk_size - overlap):
            parca = ' '.join(kelimeler[i:i + chunk_size])
            if len(parca.strip()) > 50:
                parcalar.append(parca.strip())
        
        return parcalar
    
    def add_documents_to_all(self, documents):
        """
        Belgeleri tüm indexlere ekle
        
        Args:
            documents: [{id, filename, filepath, full_text}, ...]
        """
        print("📝 Belgeler tüm indexlere ekleniyor...")
        start_time = time.time()
        
        # Metinleri parçala
        tum_metinler = []
        tum_idler = []
        tum_metadatalar = []
        embeddings_list = []
        
        for doc in documents:
            doc_id = doc.get("id", 0)
            filename = doc.get("filename", "")
            filepath = doc.get("filepath", "")
            full_text = doc.get("full_text", "")
            
            if not full_text or len(full_text.strip()) < 50:
                print(f"⚠ Atlanan döküman (boş veya çok kısa): {filename}")
                continue
            
            # Metni parçala
            parcalar = self.metni_parcala(full_text, chunk_size=300, overlap=50)
            
            for chunk_idx, parca in enumerate(parcalar):
                # ChromaDB verisi
                tum_metinler.append(parca)
                tum_idler.append(f"doc_{doc_id}_chunk_{chunk_idx}")
                tum_metadatalar.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "filepath": filepath,
                    "chunk_id": chunk_idx
                })
                
                # Embedding hesapla (tüm indexler için)
                embedding = self.model.encode(parca)
                embeddings_list.append(embedding)
                
                # Metadata sakla
                self.documents_metadata[self.counter] = {
                    'chunk_id': f"doc_{doc_id}_chunk_{chunk_idx}",
                    'text': parca,
                    'filename': filename,
                    'doc_id': doc_id
                }
                
                self.counter += 1
        
        print(f"✓ Toplam {len(tum_metinler)} adet metin parçası oluşturuldu\n")
        
        # ChromaDB'ye batch halinde ekle
        batch_size = 100
        total_added = 0
        
        print("📥 ChromaDB'ye veri ekleniyor...")
        chroma_start = time.time()
        
        for i in range(0, len(tum_metinler), batch_size):
            batch_metinler = tum_metinler[i:i+batch_size]
            batch_idler = tum_idler[i:i+batch_size]
            batch_metadatalar = tum_metadatalar[i:i+batch_size]
            
            try:
                self.collection.add(
                    documents=batch_metinler,
                    metadatas=batch_metadatalar,
                    ids=batch_idler
                )
                total_added += len(batch_metinler)
                print(f"  ✓ {total_added}/{len(tum_metinler)} parça eklendi...")
            except Exception as e:
                print(f"  ✗ ChromaDB hatası: {e}")
        
        chroma_time = time.time() - chroma_start
        print(f"✓ ChromaDB'ye {total_added} belge eklendi ({chroma_time:.2f}s)")
        
        # Embeddings numpy array'e dönüştür
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        
        # HNSW'ye ekle
        print("\n📥 HNSW index'ine veri ekleniyor...")
        hnsw_start = time.time()
        if len(embeddings_array) > 0:
            hnsw_ids = np.arange(len(embeddings_array))
            self.hnsw_index.add_items(embeddings_array, hnsw_ids)
            print(f"✓ HNSW index'ine {len(embeddings_array)} belge eklendi")
        hnsw_time = time.time() - hnsw_start
        
        # Flat'e ekle (tüm embeddingleri sakla)
        print("\n📥 Flat (Exhaustive) index'ine veri ekleniyor...")
        flat_start = time.time()
        self.flat_embeddings = embeddings_array
        self.flat_ids = np.arange(len(embeddings_array))
        print(f"✓ Flat index'ine {len(embeddings_array)} belge eklendi")
        flat_time = time.time() - flat_start
        
        # Annoy'a ekle
        print("\n📥 Annoy index'ine veri ekleniyor...")
        annoy_start = time.time()
        for idx, emb in enumerate(embeddings_array):
            self.annoy_index.add_item(idx, emb)
        self.annoy_index.build(10)  # 10 ağaç
        print(f"✓ Annoy index'ine {len(embeddings_array)} belge eklendi")
        annoy_time = time.time() - annoy_start
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✓ Toplam işlem zamanı: {total_time:.2f}s")
        print(f"  - ChromaDB: {chroma_time:.2f}s")
        print(f"  - HNSW: {hnsw_time:.2f}s")
        print(f"  - Flat: {flat_time:.2f}s")
        print(f"  - Annoy: {annoy_time:.2f}s")
        print(f"{'='*60}")
        
        # Koleksiyon bilgisi
        print(f"\n📊 Index Bilgisi:")
        print(f"   ChromaDB kayıt: {self.collection.count()}")
        print(f"   HNSW kayıt: {self.hnsw_index.get_current_count()}")
        print(f"   Flat kayıt: {len(self.flat_embeddings)}")
        print(f"   Annoy kayıt: {self.annoy_index.get_n_items()}")
    
    def search_with_flat(self, query, k=5):
        """
        Flat (Exhaustive/Doğrusal Tarama) - %100 doğruluk, yavaş
        
        Args:
            query: Arama sorgusu
            k: Döndürülecek sonuç sayısı
        """
        query_embedding = self.model.encode(query)
        
        start_time = time.time()
        
        # Tüm embeddingleri sorguyla karşılaştır (cosine similarity)
        similarities = []
        for idx, embedding in enumerate(self.flat_embeddings):
            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((idx, similarity))
        
        # Benzerliğe göre sırala
        similarities.sort(key=lambda x: x[1], reverse=True)
        search_time = time.time() - start_time
        
        # Sonuçları formatla
        results = []
        for rank, (idx, similarity) in enumerate(similarities[:k], 1):
            doc_info = self.documents_metadata[idx]
            results.append({
                'rank': rank,
                'similarity': float(similarity),
                'text': doc_info['text'],
                'filename': doc_info['filename'],
                'doc_id': doc_info['doc_id']
            })
        
        return {
            'query': query,
            'method': 'Flat (Exhaustive)',
            'search_time': search_time,
            'results': results
        }
    
    def search_with_hnsw(self, query, k=5, ef=200):
        """
        HNSW - Çok hızlı, yüksek recall
        
        Args:
            query: Arama sorgusu
            k: Döndürülecek sonuç sayısı
            ef: Arama parametresi (yüksek = daha doğru ama yavaş)
        """
        query_embedding = self.model.encode(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        self.hnsw_index.set_ef(ef)
        start_time = time.time()
        labels, distances = self.hnsw_index.knn_query(query_vector, k=k)
        search_time = time.time() - start_time
        
        results = []
        for rank, (hnsw_id, distance) in enumerate(zip(labels[0], distances[0]), 1):
            hnsw_id = int(hnsw_id)
            if hnsw_id in self.documents_metadata:
                doc_info = self.documents_metadata[hnsw_id]
                results.append({
                    'rank': rank,
                    'similarity': float(1 - distance),
                    'text': doc_info['text'],
                    'filename': doc_info['filename'],
                    'doc_id': doc_info['doc_id']
                })
        
        return {
            'query': query,
            'method': 'HNSW',
            'search_time': search_time,
            'results': results
        }
    
    def search_with_annoy(self, query, k=5):
        """
        Annoy - Ağaç tabanlı, hızlı, düşük recall
        
        Args:
            query: Arama sorgusu
            k: Döndürülecek sonuç sayısı
        """
        query_embedding = self.model.encode(query)
        
        start_time = time.time()
        neighbors, distances = self.annoy_index.get_nns_by_vector(
            query_embedding, k, include_distances=True
        )
        search_time = time.time() - start_time
        
        results = []
        for rank, (idx, distance) in enumerate(zip(neighbors, distances), 1):
            if idx in self.documents_metadata:
                doc_info = self.documents_metadata[idx]
                # Annoy angular distance'ı similarity'ye dönüştür
                similarity = 1 - (distance / 2)
                results.append({
                    'rank': rank,
                    'similarity': float(similarity),
                    'text': doc_info['text'],
                    'filename': doc_info['filename'],
                    'doc_id': doc_info['doc_id']
                })
        
        return {
            'query': query,
            'method': 'Annoy',
            'search_time': search_time,
            'results': results
        }
    
    def search_with_chromadb(self, query, n_results=5):
        """ChromaDB ile arama yap"""
        start_time = time.time()
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        search_time = time.time() - start_time
        
        formatted_results = []
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            formatted_results.append({
                'rank': i,
                'similarity': float(1 - dist),
                'text': doc,
                'filename': meta.get('filename'),
                'doc_id': meta.get('doc_id')
            })
        
        return {
            'query': query,
            'method': 'ChromaDB',
            'search_time': search_time,
            'results': formatted_results
        }


def compare_algorithms(db, query, k=5):
    """Tüm algoritmaları karşılaştır"""
    print(f"\n{'='*80}")
    print(f"📌 SORGU: '{query}'")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    # 1. Flat Search
    print("🔍 1. Flat (Exhaustive Tarama) - %100 doğruluk, yavaş")
    print("-" * 80)
    flat_result = db.search_with_flat(query, k=k)
    all_results['flat'] = flat_result
    print(f"⏱️  Arama zamanı: {flat_result['search_time']*1000:.4f}ms\n")
    
    for result in flat_result['results'][:3]:
        print(f"  {result['rank']}. (Benzerlik: {result['similarity']:.4f})")
        print(f"     📄 {result['filename']}")
        print(f"     📝 {result['text'][:80]}...\n")
    
    # 2. HNSW
    print("\n🔍 2. HNSW (Graf Tabanlı) - Hızlı, Yüksek Recall")
    print("-" * 80)
    hnsw_result = db.search_with_hnsw(query, k=k, ef=200)
    all_results['hnsw'] = hnsw_result
    print(f"⏱️  Arama zamanı: {hnsw_result['search_time']*1000:.4f}ms\n")
    
    for result in hnsw_result['results'][:3]:
        print(f"  {result['rank']}. (Benzerlik: {result['similarity']:.4f})")
        print(f"     📄 {result['filename']}")
        print(f"     📝 {result['text'][:80]}...\n")
    
    # 3. Annoy
    print("\n🔍 3. Annoy (Ağaç Tabanlı) - Çok Hızlı, Düşük Recall")
    print("-" * 80)
    annoy_result = db.search_with_annoy(query, k=k)
    all_results['annoy'] = annoy_result
    print(f"⏱️  Arama zamanı: {annoy_result['search_time']*1000:.4f}ms\n")
    
    for result in annoy_result['results'][:3]:
        print(f"  {result['rank']}. (Benzerlik: {result['similarity']:.4f})")
        print(f"     📄 {result['filename']}")
        print(f"     📝 {result['text'][:80]}...\n")
    
    # 4. ChromaDB
    print("\n🔍 4. ChromaDB - Yüksek Doğruluk")
    print("-" * 80)
    chroma_result = db.search_with_chromadb(query, n_results=k)
    all_results['chromadb'] = chroma_result
    print(f"⏱️  Arama zamanı: {chroma_result['search_time']*1000:.4f}ms\n")
    
    for result in chroma_result['results'][:3]:
        print(f"  {result['rank']}. (Benzerlik: {result['similarity']:.4f})")
        print(f"     📄 {result['filename']}")
        print(f"     📝 {result['text'][:80]}...\n")
    
    # Performans Özeti
    print(f"\n{'='*80}")
    print("⚡ PERFORMANS ÖZETİ")
    print(f"{'='*80}")
    print(f"{'Algoritma':<20} {'Arama Zamanı (ms)':<20} {'Benzerlik (Top-1)':<20}")
    print("-" * 80)
    
    for method in ['flat', 'hnsw', 'annoy', 'chromadb']:
        result = all_results[method]
        search_time = result['search_time'] * 1000
        top_sim = result['results'][0]['similarity'] if result['results'] else 0
        print(f"{result['method']:<20} {search_time:<20.4f} {top_sim:<20.4f}")
    
    print(f"{'='*80}\n")


def main():
    """Ana fonksiyon"""
    # Veritabanını başlat
    db = ChromaDBWithMultipleIndexes(
        chroma_path="./multi_index_chroma_db",
        collection_name="dokumanlarim"
    )
    
    # JSON dosyasını oku
    json_dosya_yolu = '/home/ugo/Documents/Python/bitirememe projesi/metin_dosyasi.json'
    
    if not os.path.exists(json_dosya_yolu):
        print(f"✗ HATA: {json_dosya_yolu} dosyası bulunamadı!")
        return
    
    with open(json_dosya_yolu, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Dökümanları çıkar
    documents = data.get("documents", [])
    print(f"✓ Toplam {len(documents)} adet döküman bulundu\n")
    
    # Belgeleri ekle
    db.add_documents_to_all(documents)
    
    # Test aramaları
    print("\n" + "="*80)
    print("ALGORİTMA KARŞILAŞTIRMASI")
    print("="*80)
    
    test_queries = [
        "artificial intelligence healthcare",
        "machine learning medical diagnosis",
        "deep learning neural networks"
    ]
    
    for query in test_queries:
        compare_algorithms(db, query, k=5)


if __name__ == "__main__":
    main()