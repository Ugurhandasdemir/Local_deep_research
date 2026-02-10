# Vector Database Benchmark Projesi

Bu proje, farklı vektör veritabanlarında (Milvus, Qdrant, ChromaDB, LanceDB, Weaviate, pgvector) tekli ve çoklu model senaryolarında yazma/arama performanslarını ölçmek için hazırlanmış kapsamlı bir benchmark paketidir.

## Özellikler
- Çoklu model embedding üretimi ve karşılaştırmalı benchmark
- Veritabanlarına toplu yazma ve çoklu arama senaryoları
- Sonuçların JSON/Excel olarak dışa aktarımı
- Örnek sorgular ve test scriptleri

## Kurulum
1. Gerekli Python bağımlılıklarını kurun (ör. `pip install -r requirements.txt`).
2. İlgili veritabanlarını yerel olarak çalıştırın:
   - Milvus, Qdrant, Weaviate vb.
3. Modeller ve veri dosyaları için klasörleri hazırlayın:
   - `models/`
   - `metin_dosyasi.json` veya `metin_dosyasi.txt`

## Kullanım

### Çoklu Model Benchmark
```
python write_vector_database.py
```

### Algoritma Bazlı Benchmark
```
python src/benchmark.py
```

### Weaviate / pgvector / Milvus / ChromaDB / LanceDB Yazma Scriptleri
İlgili klasörlerden çalıştırabilirsiniz:
- `DB/weaviate/write_vector_database.py`
- `DB/pgvector/write_vector_database.py`
- `DB/milvus/write_vector_database.py`
- `DB/chorame/write_vector_database.py`
- `DB/lanceDatabase/main.py`

### Sorgu Örnekleri
- `DB/weaviate/sorgu.py`
- `DB/pgvector/sorgu.py`
- `DB/chorame/sorgu.py`

## Çıktılar
- `multi_model_benchmark_results.json`
- `multi_model_benchmark_results.xlsx`
- `algorithm_benchmark.json`
- `algorithm_benchmark.xlsx`
- `algorithm_benchmark_detailed.xlsx`

## Notlar
- Veri bulunamazsa örnek veri otomatik üretilir.
- Büyük veri ve çoklu model kullanımı yüksek kaynak tüketebilir.
- Bazı veritabanları servis olarak çalıştırılmalıdır.
