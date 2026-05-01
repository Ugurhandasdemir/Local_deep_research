# Local Deep Research & Vector DB Benchmark

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Flutter](https://img.shields.io/badge/UI-Flutter-02569B)](https://flutter.dev/)
[![Status](https://img.shields.io/badge/status-research-orange)](#)

PDF dokümanlarını vektör veritabanlarına indeksleyen, yerel LLM ile soru-cevap/rapor üreten ve farklı embedding modeli + vektör veritabanı kombinasyonlarını benchmark eden bitirme projesi.

Proje iki ana parçadan oluşur:

- **Local Deep Research uygulaması:** PDF yükleme, metin çıkarma, vektör indeksleme, kaynaklı cevap üretimi ve Flutter arayüzü.
- **Benchmark paketi:** 28 embedding modelini Milvus, Qdrant, ChromaDB, LanceDB ve Weaviate üzerinde yazma hızı, arama gecikmesi ve retrieval kalitesi açısından karşılaştırır.

## İçindekiler

- [Özellikler](#özellikler)
- [Mimari](#mimari)
- [Teknolojiler](#teknolojiler)
- [Kurulum](#kurulum)
- [Dış Servisler](#dış-servisler)
- [Backend Çalıştırma](#backend-çalıştırma)
- [Flutter Arayüzü](#flutter-arayüzü)
- [Benchmark Kullanımı](#benchmark-kullanımı)
- [Veri ve Model Dosyaları](#veri-ve-model-dosyaları)
- [API Uçları](#api-uçları)
- [Benchmark Çıktıları](#benchmark-çıktıları)
- [Proje Yapısı](#proje-yapısı)
- [Sorun Giderme](#sorun-giderme)
- [Katkı](#katkı)
- [Lisans](#lisans)

## Özellikler

- PDF yükleme ve metin çıkarma
- PDF içeriğini chunk'lara ayırıp vektör indekslerine yazma
- ChromaDB, Milvus ve Weaviate üzerinde uygulama içi arama senaryoları
- Ollama üzerinden yerel LLM ile cevap/rapor üretimi
- Flutter tabanlı masaüstü/mobil arayüz
- Çoklu vektör veritabanı benchmark akışı
- Subprocess izolasyonlu benchmark worker yapısı
- Embedding cache ile tekrar encode maliyetini azaltma
- JSON ve Excel formatında benchmark sonucu üretimi

## Mimari

```text
Flutter UI
   |
   v
FastAPI Backend (main.py)
   |
   +--> PDF kaydetme ve metin çıkarma
   +--> vector_indexes.py
   |      +--> ChromaDB
   |      +--> Milvus
   |      +--> Weaviate
   |
   +--> Ollama LLM

benchmark.py
   |
   +--> model encode worker
   +--> write/search worker
   +--> quality benchmark worker
   +--> multi_model_benchmark_results.json / .xlsx
```

## Teknolojiler

- Python 3.12
- FastAPI, Uvicorn
- Flutter
- SentenceTransformers, Transformers, PyTorch
- LangChain Ollama
- ChromaDB, Milvus Lite, Qdrant, LanceDB, Weaviate
- PyPDF2, pdfplumber
- pandas, NumPy, openpyxl

## Kurulum

```bash
git clone <repo-url>
cd "bitirememe projesi"

python3.12 -m venv env
source env/bin/activate

pip install -r requirements.txt
```

Uygulamada kullanılan Ollama modelleri:

```bash
ollama pull nemotron-3-nano:4b
ollama pull medgemma1.5:latest
ollama pull granite4.1:3b
ollama pull translategemma:4b
ollama pull ministral-3:3b
```

## Dış Servisler

Uygulamadaki `balanced` senaryosu Weaviate kullanır. Benchmark tarafında Qdrant ve Weaviate için servislerin açık olması gerekir.

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

```bash
docker run -d --name weaviate \
  -p 8080:8080 \
  -p 50051:50051 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=none \
  cr.weaviate.io/semitechnologies/weaviate:latest
```

ChromaDB, LanceDB ve Milvus Lite bu projede dosya tabanlı çalışır; ayrı servis başlatmanız gerekmez.

## Backend Çalıştırma

```bash
source env/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend varsayılan olarak `http://127.0.0.1:8000` adresinde çalışır. Flutter istemcisi de bu adresi kullanır.

## Flutter Arayüzü

```bash
cd UI/local_deep_research_interface
flutter pub get
flutter run
```

API adresi şu dosyada tanımlıdır:

```text
UI/local_deep_research_interface/lib/services/api_services.dart
```

Farklı bir backend adresi kullanacaksanız `apiBaseUrl` değerini güncelleyin.

## Benchmark Kullanımı

Hızlı test modu 20 belge ile çalışır:

```bash
python benchmark.py --mode test
```

Tam benchmark tüm veri setini kullanır:

```bash
python benchmark.py --mode full
```

Cache ve önceki sonuçları sıfırlayarak başlatmak için:

```bash
python benchmark.py --mode test --reset
```

Benchmark akışı:

1. Belgeleri yükler ve chunk'lara ayırır.
2. Her embedding modeli için vektörleri üretir.
3. Vektörleri her veritabanına yazar.
4. Arama gecikmesini ölçer.
5. SQuAD tabanlı kalite metriklerini hesaplar.
6. Sonuçları JSON ve Excel dosyalarına kaydeder.

> Not: `benchmark.py` içinde `BASE_DIR` yerel proje yoluna göre tanımlıdır. Projeyi farklı bir dizinde çalıştırırsanız bu değeri kendi ortamınıza göre güncelleyin.

## Veri ve Model Dosyaları

Büyük dosyalar GitHub reposuna eklenmemelidir. Aşağıdaki klasörler/dosyalar yerel ortamda bulunmalıdır:

```text
CUSTOM_DATASET/metin_dosyasi.json
CUSTOM_DATASET/squad_dataset.json
models/
DB/
embeddings_cache/
logs/
```

Uygulama tarafında kullanılan senaryolar:

| Senaryo      | Veritabanı | Embedding       |
| ------------ | ----------- | --------------- |
| `fast`     | ChromaDB    | `all_mini_l6` |
| `balanced` | Weaviate    | `bge_squad`   |
| `best`     | Milvus      | `bge_squad`   |

## API Uçları

| Metot    | Uç                    | Açıklama                                          |
| -------- | ---------------------- | --------------------------------------------------- |
| `POST` | `/upload/pdf`        | PDF dosyası yükler, metni çıkarır ve indeksler |
| `POST` | `/upload/pdf/base64` | Base64 PDF yükler ve indeksler                     |
| `POST` | `/ingest/pdf`        | Hazır PDF metnini indeksler                        |
| `POST` | `/ask/question/ai`   | Vektör arama + LLM ile kaynaklı cevap üretir     |
| `POST` | `/normal/chat`       | Vektör arama olmadan normal sohbet/özet üretir   |
| `GET`  | `/pdfs/{filename}`   | Yüklenen PDF dosyasını döndürür               |

Örnek soru isteği:

```bash
curl -X POST http://127.0.0.1:8000/ask/question/ai \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Bu dokümanın ana konusu nedir?",
    "model": "nemotron-3-nano:4b",
    "scenario": "balanced"
  }'
```

## Benchmark Çıktıları

| Dosya/Klasör                          | Açıklama                                 |
| -------------------------------------- | ------------------------------------------ |
| `multi_model_benchmark_results.json` | Ham benchmark sonuçları ve resume durumu |
| `multi_model_benchmark_results.xlsx` | Excel raporu                               |
| `embeddings_cache/`                  | Üretilmiş embedding cache dosyaları     |
| `logs/`                              | Orchestrator ve worker logları            |
| `.worker_result_*.json`              | Geçici worker sonuç dosyaları           |

Ölçülen temel metrikler:

- Yazma süresi ve throughput
- Ortalama, p50, p95 ve p99 arama gecikmesi
- nDCG@10
- Recall@10 ve Recall@100
- MRR@10
- Hit@1 ve Hit@10
- RAM/VRAM kullanım zirveleri

## Proje Yapısı

```text
.
├── benchmark.py                         # Çoklu model/veritabanı benchmark scripti
├── main.py                              # Ana FastAPI backend
├── vector_indexes.py                    # Uygulama içi vektör indeksleme/arama yardımcıları
├── requirements.txt                     # Python bağımlılıkları
├── CUSTOM_DATASET/                      # Yerel veri setleri
├── train_models/                        # Model fine-tune notebook'ları
├── UI/
│   ├── backend/                         # Eski/basit FastAPI örneği
│   └── local_deep_research_interface/   # Flutter arayüzü
├── multi_model_benchmark_results.json   # Benchmark sonucu
├── multi_model_benchmark_results.xlsx   # Benchmark Excel raporu
└── pdfs.db                              # Yüklenen PDF içerikleri için SQLite veritabanı
```

## Katkı

Bu proje aşağıdaki ekip üyeleri tarafından geliştirilmiştir:

| İsim                | GitHub                                              |
| -------------------- | --------------------------------------------------- |
| Uğurhan Daşdemir   | [@Ugurhandasdemir](https://github.com/Ugurhandasdemir) |
| Yüksel Erhan Turgut | [@Yukseltt](https://github.com/Yukseltt)               |
| Enes Hakan Demir     | [@hakanenesdemir](https://github.com/hakanenesdemir)   |


## Lisans

Bu repoda henüz bir `LICENSE` dosyası bulunmuyor. GitHub'da açık kaynak olarak yayınlamadan önce uygun lisansı ekleyin.
