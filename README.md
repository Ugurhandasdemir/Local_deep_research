# Local Deep Research Interface

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Flutter](https://img.shields.io/badge/UI-Flutter-02569B)](https://flutter.dev/)
[![Status](https://img.shields.io/badge/status-research-orange)](#)

PDF dokümanlarını vektör veritabanlarına indeksleyen, **tamamen yerel** çalışan LLM ile soru-cevap/rapor üreten ve 26 farklı embedding modeli × 5 vektör veritabanı kombinasyonunu kapsamlı benchmark eden bitirme projesi.

**Veri dışarı çıkmaz.** Tüm model ve veritabanları makinenizde çalışır.

---

## Ekran Görüntüleri

| Ana Ekran — Senaryo Seçimi | Yanıt + Kaynak Gösterimi | Sohbet Geçmişi + Dark Mode |
|---|---|---|
| ![Ana Ekran](image/Screenshot%20from%202026-06-18%2015-51-17.png) | ![Yanıt](image/Screenshot%20from%202026-06-18%2015-55-17.png) | ![Geçmiş](image/Screenshot%20from%202026-06-18%2015-55-28.png) |

---

## İçindekiler

- [Özellikler](#özellikler)
- [Mimari](#mimari)
- [Senaryo Seçimi](#senaryo-seçimi)
- [Teknolojiler](#teknolojiler)
- [Kurulum](#kurulum)
- [Dış Servisler](#dış-servisler)
- [Backend Çalıştırma](#backend-çalıştırma)
- [Flutter Arayüzü](#flutter-arayüzü)
- [API Uçları](#api-uçları)
- [Embedding Modelleri](#embedding-modelleri)
- [Benchmark Kullanımı](#benchmark-kullanımı)
- [Benchmark Sonuçları](#benchmark-sonuçları)
- [Proje Yapısı](#proje-yapısı)
- [Teknik Notlar](#teknik-notlar)
- [Katkı](#katkı)

---

## Özellikler

- PDF yükleme ve metin çıkarma (multipart, base64, ham metin)
- PDF içeriğini chunk'lara ayırıp ChromaDB / Milvus / Weaviate'e indeksleme
- Senaryo bazlı arama: **Hızlı** / **Dengeli** / **En Yüksek Başarım**
- Ollama üzerinden yerel LLM ile Türkçe kaynaklı rapor üretimi
- Her yanıtta kaynak PDF ve benzerlik skoru gösterimi
- Flutter tabanlı masaüstü arayüz (Dark mode, sohbet geçmişi)
- 26 model × 5 DB benchmark — subprocess izolasyonu ile 4GB VRAM'de güvenli çalışma
- Embedding cache (`embeddings_cache/*.npy`) — bir kez encode, defalarca kullan
- JSON + Excel benchmark raporu, resume desteği (yarıda kalan tamamlanır)
- BEIR-style kalite metrikleri: nDCG@10, Recall@10/100, MRR@10, Hit@1/10

---

## Mimari

```
Flutter UI
    │
    ▼
FastAPI Backend  (main.py)
    │
    ├── PDF kaydetme (SQLite + dosya sistemi)
    ├── Metin çıkarma (PyPDF2)
    │
    ├── vector_indexes.py
    │       ├── ChromaDB   (fast senaryosu)
    │       ├── Weaviate   (balanced senaryosu)
    │       └── Milvus     (best senaryosu)
    │
    └── Ollama LLM  ──► Türkçe Rapor

─────────────────────────────────────────
benchmark.py  (ayrı süreç, üretim dışı)
    │
    ├── Orchestrator: model sırası yönetimi, resume
    ├── Encode Worker:   embedding hesapla → .npy'a yaz
    ├── Write Worker:    .npy'dan oku → DB'ye yaz
    ├── Search Worker:   QPS / p50 / p95 ölç
    └── Quality Worker:  SQuAD üzerinde nDCG@10, MRR@10
```

---

## Senaryo Seçimi

Üç hazır yapılandırma — hız/kalite dengesi:

| Senaryo | Arayüz Adı | DB | Embedding | Boyut | Kullanım |
|---|---|---|---|---|---|
| `fast` | Hızlı | ChromaDB | all_mini_l6 | 384 | Prototip, düşük gecikme |
| `balanced` | Dengeli | Weaviate | bge_squad | 1024 | Üretim varsayılanı |
| `best` | En Yüksek Başarım | Milvus | bge_squad | 1024 | Maksimum doğruluk |

API isteğinde `db` ve `embedding` parametreleriyle senaryo dışına çıkılabilir.

---

## Teknolojiler

| Katman | Kütüphane / Araç |
|---|---|
| API | FastAPI, Uvicorn |
| LLM | LangChain-Ollama |
| Embedding | SentenceTransformers, Transformers, PyTorch |
| Vektör DB | ChromaDB, Milvus Lite, Weaviate, Qdrant, LanceDB |
| PDF | PyPDF2, pdfplumber |
| Veri | NumPy, pandas, openpyxl |
| UI | Flutter |
| Sistem | psutil, systemd-inhibit |

---

## Kurulum

```bash
git clone <repo-url>
cd "bitirememe projesi"

python3.12 -m venv env
source env/bin/activate

pip install -r requirements.txt
```

Ollama modelleri (en az birini indirin):

```bash
ollama pull nemotron-3-nano:4b    # varsayılan, düşük VRAM
ollama pull ministral-3b:latest
ollama pull granite4.1:3b
```

---

## Dış Servisler

ChromaDB, LanceDB ve Milvus Lite **dosya tabanlı** çalışır — ayrı servis gerekmez.

Qdrant ve Weaviate için Docker:

```bash
# Qdrant (benchmark için)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Weaviate (balanced senaryosu + benchmark için)
docker run -d --name weaviate \
  -p 8080:8080 -p 50051:50051 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=none \
  cr.weaviate.io/semitechnologies/weaviate:latest
```

---

## Backend Çalıştırma

```bash
source env/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# veya
python main.py
```

Backend `http://127.0.0.1:8000` adresinde çalışır.

---

## Flutter Arayüzü

```bash
cd UI/local_deep_research_interface
flutter pub get
flutter run
```

Backend adresi `lib/services/api_services.dart` içindeki `apiBaseUrl` değişkeninde tanımlıdır.

---

## API Uçları

| Metot | Uç | Açıklama |
|---|---|---|
| `POST` | `/upload/pdf` | Multipart PDF yükle, metni çıkar ve indeksle |
| `POST` | `/upload/pdf/base64` | Base64 PDF yükle ve indeksle |
| `POST` | `/ingest/pdf` | Hazır PDF metnini indeksle |
| `POST` | `/ask/question/ai` | Vektör arama + LLM → kaynaklı Türkçe rapor |
| `POST` | `/normal/chat` | Vektörsüz normal sohbet / özet |
| `GET` | `/pdfs/{filename}` | Yüklenen PDF dosyasını getir |

### Soru Sor

```bash
curl -X POST http://127.0.0.1:8000/ask/question/ai \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Yapay zeka sağlıkta nasıl kullanılır?",
    "model": "nemotron-3-nano:4b",
    "scenario": "balanced"
  }'
```

**Yanıt:**
```json
{
  "status": "success",
  "aiResponse": "Tıp alanında AI...",
  "sources": [
    {
      "file": "056_artificial-intelligence-in-healthcare.pdf",
      "url": "http://127.0.0.1:8000/pdfs/056_artificial...",
      "score": 0.8741,
      "metin": "..."
    }
  ]
}
```

---

## Embedding Modelleri

### Yerel Fine-Tuned (`models/` klasörü)

| Anahtar | Temel Model | Boyut |
|---|---|---|
| `all_mini_l6` | all-MiniLM-L6-v2 | 384 |
| `bge_squad` | BGE-large-en-v1.5 (SQuAD fine-tune) | 1024 |
| `e5_small` | multilingual-e5-small | 384 |
| `e5_base` | multilingual-e5-base | 768 |
| `mpnet_multi` | paraphrase-multilingual-mpnet-base-v2 | 768 |
| `qwen_lora` | Qwen3-Embedding-0.6B (LoRA) | 1024 |
| `snowflake_arctic_l` | Snowflake/snowflake-arctic-embed-l-v2.0 | 1024 |
| `bge_m3_fine` | BAAI/bge-m3 | 1024 |

### HuggingFace Base Modeller (benchmark için)

`e5_small_base`, `mpnet_multi_base`, `e5_base_base`, `bge_squad_base`, `qwen_lora_base`, `snowflake_arctic_l_base`, `all_mini_l6_base`, `bge_m3_base`, `minilm_l12`, `mpnet_base`, `distilroberta`, `multi_qa_minilm`, `multi_qa_mpnet`, `paraphrase_multi`, `bge_small_en`, `bge_base_en`, `gte_small`, `gte_base`, `e5_small_hf`, `e5_base_hf`

---

## Benchmark Kullanımı

26 embedding modeli × 5 vektör veritabanı. **4GB VRAM** kısıtı altında çalışır.

```bash
# Hızlı test (20 belge)
python benchmark.py --mode test

# Tam benchmark (52.331 belge)
python benchmark.py --mode full

# Sıfırdan başlat (cache + sonuçları sil)
python benchmark.py --mode full --reset
```

**Benchmark akışı:**

1. **Encode** — her model için belgeler ve sorgular encode edilir, `.npy`'a kaydedilir
2. **Write** — her (model, DB) için belgeler veritabanına yazılır, süre ölçülür
3. **Search** — 1000 sorgu × 3 tur ile QPS/gecikme ölçülür
4. **Encode Qual** — SQuAD korpusu encode edilir
5. **Write Qual** — SQuAD vektörleri DB'ye yazılır
6. **Search Qual** — nDCG@10, Recall@10/100, MRR@10, Hit@1/10 hesaplanır

Her aşama ayrı subprocess — aşama bitince VRAM/RAM tamamen serbest bırakılır.
Yarıda kalan benchmark `--reset` olmadan yeniden başlatılınca tamamlananlar atlanır.

---

## Benchmark Sonuçları

**Ortam:** full mod, 52.331 belge, 1000 sorgu (SQuAD), 4GB VRAM

### En Hızlı Arama (ms, ortalama)

| Sıra | DB | Model | Algoritma | Ort (ms) |
|---|---|---|---|---|
| 1 | ChromaDB | minilm_l12 | HNSW_batch | **2.955** |
| 2 | ChromaDB | all_mini_l6_base | HNSW_batch | 3.187 |
| 3 | ChromaDB | gte_small | HNSW_batch | 3.268 |
| 4 | ChromaDB | bge_small_en | HNSW_batch | 3.492 |
| 5 | ChromaDB | multi_qa_minilm | HNSW_batch | 3.501 |
| 80 | Milvus | e5_small | HNSW_batch | 9.734 |
| 234 | Qdrant | mpnet_multi | HNSW_ef32 | 19.555 |

### Yazma Hızı (52.331 kayıt, saniye)

| DB | En Hızlı Model | Süre | En Yavaş Model | Süre |
|---|---|---|---|---|
| Milvus | e5_small | 9.5s | bge_squad | 23.6s |
| Weaviate | minilm_l12 | 10.8s | snowflake_arctic_l | 17.5s |
| Qdrant | e5_small | 17.9s | bge_squad | 47.0s |
| LanceDB | gte_base | 38.7s | snowflake_arctic_l | 40.7s |
| ChromaDB | minilm_l12 | 87.8s | bge_squad | 97.1s |

**Özet:** Milvus yazma hızında lider; ChromaDB yazma en yavaş fakat arama en hızlı (HNSW_batch).

### Senaryo Önerileri (benchmark bulgularına göre)

| Öncelik | Tercih |
|---|---|
| En hızlı arama | ChromaDB + minilm_l12 (2.9ms) |
| En hızlı yazma | Milvus + e5_small (9.5s) |
| Hız/kalite dengesi | Weaviate + bge_squad (balanced senaryosu) |
| Hybrid arama | Weaviate (BM25 + vektör, alpha=0.5) |
| Tam doğruluk | Milvus + bge_squad (best senaryosu) |

> `qwen_lora`, `bge_m3_fine`, `bge_m3_base` modelleri 4GB VRAM'de OOM nedeniyle tamamlanamadı.

---

## Proje Yapısı

```
bitirememe projesi/
├── main.py                              # FastAPI backend
├── vector_indexes.py                    # Uygulama vektör indeks + arama
├── benchmark.py                         # 26 model × 5 DB benchmark
├── requirements.txt
├── pdfs.db                              # SQLite: yüklenen PDF ikilileri
├── multi_model_benchmark_results.json   # Ham benchmark sonuçları
├── multi_model_benchmark_results.xlsx   # Excel raporu
├── multi_model_benchmark_results.md     # Markdown raporu
├── embeddings_cache/                    # Önceden hesaplanmış vektörler (.npy)
├── models/                              # Yerel fine-tuned embedding modelleri
├── DB/                                  # Vektör DB verileri
│   ├── chromadb/
│   ├── milvus/
│   ├── weaviate/ (Docker volume)
│   ├── qdrant/   (Docker volume)
│   └── lancedb/
├── data/dataset_pdf/                    # Yüklenen PDF dosyaları
├── CUSTOM_DATASET/
│   ├── metin_dosyasi.json               # Benchmark belge seti
│   └── squad_dataset.json               # SQuAD kalite değerlendirme seti
├── logs/                                # Orchestrator + worker logları
├── image/                               # Ekran görüntüleri
└── UI/
    └── local_deep_research_interface/   # Flutter arayüzü
```

---

## Teknik Notlar

| Parametre | Değer | Açıklama |
|---|---|---|
| Chunk boyutu | 500 karakter | 50 karakter örtüşme ile kayan pencere |
| LLM context | 2048 token | Varsayılan 32k'dan düşürüldü — 4GB VRAM |
| LLM max yanıt | 512 token | `num_predict` sınırı |
| LLM sıcaklık | 0 (DR) / 0.7 (sohbet) | Deep Research deterministik |
| ChromaDB mesafe | cosine | HNSW indexi |
| Milvus metric | COSINE | |
| Milvus ID | SHA1-tabanlı int64 | `1e12 + sha1[:8] % 8e18` |
| Weaviate ID | UUID5 | `uuid5(NAMESPACE_URL, key:file:idx)` |
| OOM koruması | batch yarıya indir | CUDA OOM → `batch//2` → son çare CPU |
| Encode cache | `.npy` dosyaları | Tekrar çalıştırmada encode yok |
| Benchmark izolasyon | subprocess | Her (model, DB) ayrı süreç → tam VRAM temizleme |
| Suspend engeli | systemd-inhibit | Benchmark sırasında uyku/kapak kilidi |

---

## Katkı

| İsim | GitHub |
|---|---|
| Uğurhan Daşdemir | [@Ugurhandasdemir](https://github.com/Ugurhandasdemir) |
| Yüksel Erhan Turgut | [@Yukseltt](https://github.com/Yukseltt) |
| Enes Hakan Demir | [@hakanenesdemir](https://github.com/hakanenesdemir) |
