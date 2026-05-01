# Vektör Veritabanı Çoklu-Model Benchmark Paketi

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#lisans)
[![Status](https://img.shields.io/badge/status-research-orange.svg)](#)

5 farklı vektör veritabanı (**Milvus**, **Qdrant**, **ChromaDB**, **LanceDB**, **Weaviate**) üzerinde **28 embedding modelini** akademik standartlara uygun (BEIR/MTEB-style) şekilde karşılaştıran, sınırlı GPU/RAM ortamlarında güvenli çalışan bir benchmark çatısı.

---

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Özellikler](#özellikler)
- [Mimari](#mimari)
- [Ölçülen Metrikler](#ölçülen-metrikler)
- [Desteklenen Modeller](#desteklenen-modeller)
- [Sistem Gereksinimleri](#sistem-gereksinimleri)
- [Kurulum](#kurulum)
- [Veritabanı Servislerini Başlatma](#veritabanı-servislerini-başlatma)
- [Kullanım](#kullanım)
- [Çıktılar](#çıktılar)
- [Resume / Devam Mekanizması](#resume--devam-mekanizması)
- [Proje Yapısı](#proje-yapısı)
- [Sorun Giderme](#sorun-giderme)
- [Yol Haritası](#yol-haritası)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)
- [Atıf](#atıf)

---

## Genel Bakış

Bu proje, kurumsal RAG / semantic search senaryolarında **doğru embedding modeli + doğru vektör veritabanı** seçimini veriye dayalı yapmak için tasarlandı. Geleneksel benchmark'ların aksine:

- **Subprocess izolasyonu** ile 4 GB VRAM'de bile 28 modeli sırayla çalıştırır — bellek sızıntısı veya CUDA OOM tüm benchmark'ı düşürmez.
- **Akademik kalite metrikleri** (nDCG@10, Recall@10/100, MRR@10) SQuAD korpusu üzerinde BEIR-style ölçer.
- **Resume desteği** — yarıda kalan benchmark kaldığı yerden devam eder.
- **Disk-cache** — embedding'ler `.npy` olarak saklanır; aynı model bir daha encode edilmez.

## Özellikler

- ✅ **5 vektör DB** — Milvus, Qdrant, ChromaDB, LanceDB, Weaviate
- ✅ **28 embedding modeli** — bge-m3, e5, mpnet, gte, snowflake-arctic, qwen3-embedding, fine-tuned varyantlar
- ✅ **3 fazlı benchmark** — Yazma hızı / Arama gecikmesi / Bulma başarımı (kalite)
- ✅ **Akademik metrikler** — nDCG@k, Recall@k, MRR@10, Hit@k (BEIR uyumlu)
- ✅ **Subprocess izolasyonu** — her (model, db) kombinasyonu ayrı süreçte
- ✅ **Bellek güvenli** — RSS + VRAM peak ölçümü, OOM toleranslı (batch halving)
- ✅ **Otomatik kayıt** — her test sonrası JSON/Excel/Markdown dışa aktarım
- ✅ **Resume** — JSON-based skip kontrolü
- ✅ **Heartbeat & log** — uzun süreli çalışmalar için adım adım izlenebilir kayıt

## Mimari

```
┌──────────────────────────────────────────────────────────────┐
│                    Orchestrator (ana proses)                 │
│  - Model × DB sırasını planlar                               │
│  - Worker subprocess çağırır                                 │
│  - JSON/Excel/MD'yi her adımda günceller                     │
│  - Resume kontrolü yapar                                     │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │  Worker Subprocess (her kombinasyon için ayrı)         │
   │  Phases: encode | write | search                       │
   │          encode_qual | write_qual | search_qual        │
   └────────────────────────────────────────────────────────┘
       │            │              │            │
       ▼            ▼              ▼            ▼
  Embedding     Vector DB      Search        Quality
   Cache        (Milvus,      Latency        (nDCG,
   (.npy)       Qdrant,…)     (p50, p95)     Recall)
```

**Kritik tasarım kararları:**
- Subprocess yaklaşımı — model `del` + `torch.cuda.empty_cache()` her zaman çalışmaz; süreç sonlanması garantili temizliktir.
- Cache stratejisi — embedding üretimi en pahalı adım; bir kez yapılır, write/search fazları `.npy`'dan okur.
- Batch fallback — encode OOM verirse batch yarılanır (`batch=2 → 1`); CPU'ya düşmez (kasıtlı).

## Ölçülen Metrikler

### Faz 1 — Yazma (Write)
- Toplam yazma süresi (saniye)
- Yazma throughput'u (kayıt/saniye)
- VRAM peak (GB)

### Faz 2 — Arama (Search)
- Sorgu başına ortalama gecikme (ms)
- p50, p95, p99 gecikme
- min/max gecikme
- HNSW algoritma varyantları (default, batch, n5, limit5)

### Faz 3 — Bulma Başarımı (Quality)
SQuAD korpusu üzerinde 1000 sorgu, top-100 retrieval:
- **nDCG@10** — birincil metrik (BEIR standardı)
- **Recall@10**, **Recall@100**
- **MRR@10**
- **Hit@1**, **Hit@10**

## Desteklenen Modeller

### Yerel (fine-tuned)
| Model | Boyut | Açıklama |
|---|---|---|
| `e5_small` | 384 | Multilingual E5 small, SQuAD fine-tune |
| `e5_base` | 768 | Multilingual E5 base, SQuAD fine-tune |
| `mpnet_multi` | 768 | Paraphrase multilingual MPNet, fine-tune |
| `bge_squad` | 1024 | BGE-large EN, SQuAD fine-tune |
| `bge_m3_fine` | 1024 | BGE-M3, custom fine-tune |
| `qwen_lora` | 1024 | Qwen3-Embedding 0.6B + LoRA |
| `snowflake_arctic_l` | 1024 | Snowflake Arctic-embed L v2 |
| `all_mini_l6` | 384 | all-MiniLM-L6-v2 fine-tune |

### Base (HuggingFace, eğitilmemiş karşılaştırma)
Aynı modellerin `_base` varyantları + ek HF modelleri: `minilm_l12`, `mpnet_base`, `distilroberta`, `multi_qa_minilm`, `multi_qa_mpnet`, `paraphrase_multi`, `bge_small_en`, `bge_base_en`, `gte_small`, `gte_base`, `e5_small_hf`, `e5_base_hf`, `bge_m3_base`.

## Sistem Gereksinimleri

**Minimum**
- Python 3.12
- 8 GB RAM
- 4 GB VRAM (NVIDIA GPU, CUDA 12.x)
- 20 GB disk

**Önerilen**
- 16 GB RAM
- 8 GB+ VRAM
- SSD (cache I/O için)

**Test edilmiş ortam:** Ubuntu 22.04, RTX 3050 Ti Laptop (4 GB VRAM), Python 3.12, PyTorch 2.5.1+cu124.

## Kurulum

```bash
# 1) Repo klonla
git clone <repo-url>
cd "bitirememe projesi"

# 2) Sanal ortam
python3.12 -m venv env
source env/bin/activate

# 3) Bağımlılıklar
pip install -r requirements.txt

# 4) Veri dosyası
# metin_dosyasi.json — proje köküne yerleştir (52K+ chunk)
```

## Veritabanı Servislerini Başlatma

Her DB ayrı port'ta çalışır. En kolay: Docker.

```bash
# Milvus
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest

# Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Weaviate
docker run -d --name weaviate -p 8080:8080 cr.weaviate.io/semitechnologies/weaviate:latest

# ChromaDB & LanceDB — embedded mod, servis gerekmez
```

## Kullanım

### Hızlı test (20 belge)
```bash
python write_vector_database.py --mode test
```

### Tam benchmark (tüm korpus)
```bash
python write_vector_database.py --mode full
```

### Cache + sonuçları sıfırla
```bash
python write_vector_database.py --mode full --reset
```

### Beklenen süre
- `test` modu: ~5–10 dk
- `full` modu: 4–8 saat (donanıma göre)

## Çıktılar

| Dosya | İçerik |
|---|---|
| `multi_model_benchmark_results.json` | Ham sonuçlar (resume için) |
| `multi_model_benchmark_results.xlsx` | 17 sheet'li Excel raporu |
| `multi_model_benchmark_results.md` | AI/LLM tüketimi için Markdown |
| `logs/benchmark_orchestrator_*.log` | Ana süreç logları |
| `logs/benchmark_worker_*.log` | Subprocess detay logları |
| `embeddings_cache/*.npy` | Önbelleklenmiş vektörler |

### Excel sheet'leri
- **Ozet** — Genel özet, model meta, en hızlı 10
- **Yazma Sureleri** / **Arama Sonuclari** — Tablo formatı
- **MILVUS / QDRANT / CHROMADB / LANCEDB / WEAVIATE** — DB başına detay
- **Bulma Basarimi** — nDCG/Recall özet
- **BB_*** — Her DB için kalite detayı
- **Write Throughput** / **Search QPS p99** — Türetilmiş metrikler
- **Encode VRAM** — Bellek peak'leri

## Resume / Devam Mekanizması

JSON dosyası kayıt durumunu tutar. Yeniden başlatıldığında:

```python
is_write_done(db, model)        # write[db][model].status == "ok"
is_search_done(db, model)       # search[db][model] geçerli timing var mı
is_qual_write_done(db, model)   # quality_write[db][model].status == "ok"
is_qual_search_done(db, model)  # quality_search[db][model].status == "ok"
```

Tamamlananlar atlanır; sadece eksik/hatalılar yeniden çalışır.

## Proje Yapısı

```
bitirememe projesi/
├── write_vector_database.py     # Ana benchmark script (orchestrator + worker)
├── main.py                       # FastAPI RAG demo backend
├── requirements.txt
├── metin_dosyasi.json           # Korpus
├── models/                       # Yerel fine-tuned modeller
│   ├── e5_small/
│   ├── bge-m3-fine/
│   ├── qwen_lora/
│   └── ...
├── DB/                           # Eski tek-DB script'ler (referans)
│   ├── milvus/
│   ├── qdrant/
│   ├── chromadb/
│   └── ...
├── src/
│   ├── benchmark/                # Algoritma karşılaştırmaları
│   └── models/                   # Model utilities
├── UI/
│   ├── backend/                  # FastAPI servis
│   └── local_deep_research_interface/  # Flutter UI
├── embeddings_cache/             # .npy vektör önbelleği
├── logs/                         # Çalışma logları
└── train_models/                 # Model fine-tune script'leri
```

## Sorun Giderme

**`CUDA out of memory`**
- `MODELS` dict'inde ilgili modele `"batch": 2, "max_seq_len": 128` ekle.
- Başka GPU işlemi olmadığını doğrula: `nvidia-smi`.

**`ValueError: torch.load … CVE-2025-32434`**
- Model `.bin` formatında. Ya safetensors snapshot'a yönlendir, ya `pip install -U torch>=2.6`.

**`MilvusException: Invalid collection`**
- Model adı tire (`-`) içeriyor. Milvus sadece `[a-zA-Z0-9_]` kabul eder. Model key'inde tireleri alt çizgi yap.

**`ModuleNotFoundError: No module named 'pymilvus'`**
- Sanal ortam aktif değil. `source env/bin/activate` sonra çalıştır.

**Worker process timeout**
- `logs/benchmark_worker_*.log` kontrol et, son `HEARTBEAT` satırı son canlı zamanı gösterir.

## Yol Haritası

- [ ] pgvector ve Annoy entegrasyonu
- [ ] Hibrit arama (dense + BM25) benchmark'ı
- [ ] Multi-vector / ColBERT desteği
- [ ] Otomatik HTML rapor üretimi
- [ ] Docker Compose ile tek komutta tüm DB'ler
- [ ] CI ile küçük korpus üzerinde regresyon testi

## Katkıda Bulunma

PR'lar memnuniyetle. Lütfen:

1. Issue aç, ne değişeceğini tartış
2. Fork → feature branch → PR
3. Yeni model ekleme: `MODELS` dict'ine ekle, `dim` doğru olduğunu kontrol et
4. Yeni DB ekleme: `write_*`, `search_*`, `qual_write_*`, `qual_search_*` fonksiyonları + dict kayıtları

## Lisans

MIT License — ayrıntı için `LICENSE` dosyasına bakın.

## Atıf

Bu proje akademik bir bitirme tezi kapsamında geliştirildi. Çalışmadan yararlanırsanız:

```bibtex
@misc{vector_db_benchmark_2026,
  title  = {Vektör Veritabanı Çoklu-Model Benchmark Paketi},
  author = {Daşdemir, Uğurhan},
  year   = {2026},
  note   = {SQuAD korpusu üzerinde 28 embedding modeli × 5 vektör DB karşılaştırması}
}
```

## Teşekkür

- [BEIR](https://github.com/beir-cellar/beir) — değerlendirme metodolojisi
- [MTEB](https://github.com/embeddings-benchmark/mteb) — model karşılaştırma standartları
- [SentenceTransformers](https://www.sbert.net/) — embedding altyapısı
- [HuggingFace](https://huggingface.co/) — model ekosistemi
