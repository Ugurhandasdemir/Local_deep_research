# Yapılacaklar

Bu dosyada yalnızca yapılacaklar listesi var. Eski README `docs/README_yedek_2026-11-07.md` içinde.
Bir madde eklemeden ya da değiştirmeden önce (insan ya da AI fark etmez) `makale/AI-PROTOKOL.md` dosyasını oku.

Madde biçimi:

```
- [ ] **Y-NN** [alan] Ne yapılacak, tek cümle. · ekleyen: Ad (araç) · sorumlu: — · kaynak: dosya veya link
```

Alanlar: `deney`, `makale`, `kod`, `repo`. Yeni madde en son kullanılan numaradan devam eder. Numaralar silinmez, yeniden kullanılmaz.

## Deney ve ölçüm

- [ ] **Y-01** [deney] SQuAD setini paragraf bazında train/dev/test olarak ayır (bir paragraf tek tarafta kalsın), fine-tune'u yalnız train ile yeniden yap, sonuçları yalnız test üzerinde raporla. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: makale/incelemeler/2026-11-07-ugurhan-claude.md#A1 · **ertelendi: C aşamasına (karar K-01, makale/kararlar.md)**
- [ ] **Y-02** [deney] Kaliteyi harici bir sette de ölç (BEIR'den SciFact/NFCorpus gibi), sadece kendi SQuAD setimize güvenmeyelim. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: incelemeler#A1 · **K-01 ile Y-36'ya bağlandı**
- [ ] **Y-03** [deney] Base modelleri doğru önekle yeniden encode et: BGE sorgusuna `Represent this sentence for searching relevant passages: `, E5'te `query: ` / `passage: `. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: benchmark.py encode_with_fallback
- [ ] **Y-04** [deney] Kalite ölçümünü büyük korpusta yap: SQuAD paragraflarını 52.331 chunk'ın içine göm. Şu an kalite korpusu sadece 1.059 paragraf. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: incelemeler#A2 · **ertelendi: C aşamasına (karar K-01, makale/kararlar.md)**
- [ ] **Y-05** [deney] ANN recall'u ölç (exact kNN sonucuna göre recall@10), her DB'de HNSW M ve ef_construction değerlerini eşitle, ef/nprobe taramasıyla recall–QPS eğrisi çiz. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: incelemeler#A3
- [ ] **Y-06** [deney] Hız protokolünü yeniden kur: tekil sorgu ile batch sonuçları ayrı tablolarda olsun, gömülü DB'ler (Milvus Lite, Chroma, LanceDB) ile sunucu DB'ler (Qdrant, Weaviate) ayrılsın, 1000 tekil gecikmeden gerçek p50/p95/p99 hesaplansın, en az 3 bağımsız koşunun ortalaması ± std verilsin. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: incelemeler#A4
- [ ] **Y-07** [deney] Tüm benchmark'ı tek seferde temiz bir ortamda koştur; JSON, MD, XLSX ve makaledeki tablolar bu tek koşudan otomatik üretilsin (şu anki JSON'da arama sonuçlarının hepsi hata, query_count=10). · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: incelemeler#A5
- [ ] **Y-08** [deney] Uçtan uca RAG değerlendirmesi: 100-200 soruluk altın set hazırla, RAGAS ile faithfulness, answer relevance ve context precision ölç (hakem LLM de yerel çalışsın), 3-4 yerel LLM'i karşılaştır. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: incelemeler#B
- [ ] **Y-09** [deney] BM25 ve hibrit (alpha taraması) için nDCG ölç. Makalenin "semantik arama anahtar kelimeden iyidir" iddiası şu an sayıyla desteklenmiyor. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: incelemeler#B
- [ ] **Y-10** [deney] Donanım ve sürüm tablosu çıkar: CPU, RAM, GPU, disk, OS, her DB'nin ve kütüphanenin sürümü, Docker kaynak limitleri. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: incelemeler#B
- [ ] **Y-11** [deney] Gerçek kullanıcı gecikmesini ölç: sorgu encode + arama + LLM (TTFT ve toplam süre). Şu an sorgu vektörleri önceden cache'leniyor. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: incelemeler#C
- [ ] **Y-29** [deney] Soru setini tekilleştir (1.405 birebir kopya var), birden fazla paragrafa uyan 100 soru metnini ve bağlama bağımlı soruları ("this study", "the article") ele ya da yeniden yazdır, 100-200 soruyu iki kişi elle doğrulasın. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: makale/incelemeler/2026-11-07-ugurhan-claude-kod-2.md#1 · **ertelendi: C aşamasına (karar K-01, makale/kararlar.md)**
- [ ] **Y-31** [deney] Bellek ölçümünü düzelt: şu an yalnızca istemci sürecinin iş sonundaki RSS değeri okunuyor, Qdrant/Weaviate sunucu belleği hiç ölçülmüyor. `docker stats` ve örneklemeyle gerçek tepe değeri al. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: makale/incelemeler/2026-11-07-ugurhan-claude-kod-2.md#2
- [ ] **Y-32** [deney] Yazma süresine indeksin hazır olma süresini dahil et ya da ikisini ayrı ölç (Chroma indeksi senkron kuruyor, Qdrant arka planda). · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: makale/incelemeler/2026-11-07-ugurhan-claude-kod-2.md#2
- [ ] **Y-36** [deney] Embedding kalitesini BEIR biyomedikal setlerinde (TREC-COVID, NFCorpus, SciFact) ölç. Base modellerde doğru öneki kullan (Y-03). Fine-tune modellerimizi de aynı sette karşılaştır, böylece SQuAD'a ezberleyip ezberlemediklerini görürüz. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: makale/kararlar.md#K-01
- [ ] **Y-37** [deney] DB hızı, ölçek ve ANN recall testini VectorDBBench'in Cohere 1M setine (768 boyut) taşı. Ground truth exact kNN olsun. Y-05 ve Y-06 bu sette yapılacak. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: makale/kararlar.md#K-01
- [ ] **Y-38** [deney] Uçtan uca PDF RAG değerlendirmesini UDA'nın akademik PDF kısmında yap. Bizim PDF ayrıştırma + chunk + retrieval + LLM hattımız kullanılsın, Y-08'deki metrikler bu sette ölçülsün. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: makale/kararlar.md#K-01
- [ ] **Y-39** [deney] Türkçe iddiası makalede kalacaksa TR-MTEB'in retrieval görevlerini (Scifact-TR, NFCorpus-TR, MsMarco-TR) ekle, makine çevirisi olduklarını yaz. Kalmayacaksa Türkçe iddiasını makaleden çıkar. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: makale/kararlar.md#K-01
- [ ] **Y-41** [deney] Karşılaştırılacak embedding modelleri: Qwen3-Embedding, EmbeddingGemma 2, BGE-M3, Jina Embeddings v3, Multilingual E5 Base. Her biri için Hugging Face kimliği, boyutu (Qwen3-Embedding'in 0.6B/4B/8B sürümleri var) ve lisansı yazılsın. Modelin istediği önek ya da talimat kullanılsın (E5 için `query: `/`passage: `, Qwen3 için sorgu talimatı, bkz. Y-03). Ölçüm Y-36'daki BEIR setlerinde yapılsın. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: Uğurhan'ın model listesi (2026-11-07)
- [ ] **Y-43** [deney] Karşılaştırılacak yerel LLM'ler (Ollama): Qwen3.5, Gemma 4, Ministral 3. Boyut ve quantization 4 GB VRAM'e göre seçilsin ve tabloya yazılsın. Y-08'deki RAGAS metrikleri ve TTFT ile toplam süre (Y-11) aynı soru setinde ölçülsün. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: Uğurhan'ın model listesi (2026-11-07)

## Makale

- [ ] **Y-12** [makale] Kapsamı kodla eşitle. Netleştirme sorusu, iteratif araştırma döngüsü, reranker, üretimde hibrit arama ve GPU/CPU model takası makalede anlatılıyor ama kodda yok. Ya yazalım ya makaleden çıkaralım (karar için issue açılmalı). · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: incelemeler#B
- [ ] **Y-13** [makale] Kaynakçayı metne göre baştan kur. [25] nötrino makalesine gidiyor, [15] ile [16] aynı kayıt, [20]-[24] numaraları kaymış, SQuAD ve vektör DB'ler hiç atıflanmamış. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: incelemeler#B
- [ ] **Y-14** [makale] Veri setini tanımla: 329 PDF'in kaynağı ve lisansı, SQuAD formatındaki soruların nasıl üretildiği, soru ve paragraf sayıları. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: incelemeler#B
- [ ] **Y-15** [makale] Fine-tune sürecini yaz: veri, epoch, loss, hard negative madenciliği, donanım. Sonra base ve fine-tune karşılaştırmasını held-out test üzerinden ver. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: train_models/
- [ ] **Y-16** [makale] Yanlış ya da kanıtsız iddiaları düzelt: "Milvus dağıtık mimarisi" (kullanılan Milvus Lite), "halüsinasyondan arındırılmış", "güvenliği kanıtlamıştır", "52.331 kayıtta doğruluk". · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: incelemeler#A2, #B
- [ ] **Y-17** [makale] Tablo ve rakam tutarlılığı: 5501,4 / 5510,4, LanceDB QPS hücresinde "2.Oca", 28 / 26 model, tur sayısı (kod 3, md 10, makale 5). · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: incelemeler#A5
- [ ] **Y-18** [makale] Dil ve şablon: gövde tek dilde olsun, bölüm başlıklarındaki İ harflerini düzelt ("INTRODUCTİON"), yazım hatalarını gider (Chorame, embbeding), bölüm sırasını ve tablo/şekil numaralarını düzelt. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: incelemeler#B
- [ ] **Y-19** [makale] İlgili çalışmalar bölümüne vektör DB benchmark literatürünü ekle (ANN-Benchmarks, VectorDBBench, BEIR, vektör DB anketleri) ve katkımızı bunlara göre konumlandır. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: incelemeler#E
- [ ] **Y-30** [makale] Soru setinin ministral-3:8b ile sentetik üretildiğini ve üretim protokolünü (`CUSTOM_DATASET/text_search.py`) makalede açıkça yaz. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: makale/incelemeler/2026-11-07-ugurhan-claude-kod-2.md#1 · **ertelendi: C aşamasına (karar K-01, makale/kararlar.md)**

## Kod

- [ ] **Y-20** [kod] Güvenlik: CORS `*` ile `allow_credentials` birlikte kullanılmasın, basit bir auth eklensin, yükleme boyutu sınırlansın, prod'da `reload=True` olmasın, hata mesajlarında iç yol bilgisi dönmesin. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: main.py:18-24
- [ ] **Y-21** [kod] Tek bir indeksleme hattı kalsın. `upsert_default_chroma_chunks`, `get_collection` ve `hnsw_vector_search` ölü ya da çift iş yapıyor. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: main.py:148-206
- [ ] **Y-22** [kod] PDF güncellendiğinde eski chunk'lar silinsin, Weaviate yazımı batch ile yapılsın, DB bağlantıları her aramada yeniden açılmasın. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: vector_indexes.py:230-348
- [ ] **Y-23** [kod] Retrieval: minimum benzerlik eşiği, aynı dosyadan gelen sonuçların tekilleştirilmesi, chunk metadata'sına sayfa numarası eklenmesi, bağlamın `num_ctx` içine sığdığının token bazında kontrolü. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: vector_indexes.py, main.py:113-146
- [ ] **Y-24** [kod] Hata durumunda doğru HTTP kodu dönsün (şu an her şey 200 ve `status: error`). · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: main.py
- [ ] **Y-33** [kod] OOM durumunda gerçekten CPU'ya düş, ya da `encode_with_fallback` docstring'ini ve README'yi düzelt (şu an yalnızca batch küçültülüyor). · ekleyen: Uğurhan (Claude) · sorumlu: — · benchmark.py:408-472
- [ ] **Y-34** [kod] Flutter: yüklemede `status` alanını kontrol et (hata şu an "başarılı" görünüyor), tüm yanıtlarda `utf8.decode` kullan, isteklere timeout ekle, konuşma geçmişini backend'e gönder. · ekleyen: Uğurhan (Claude) · sorumlu: — · UI/local_deep_research_interface/lib/services/api_services.dart:81-90
- [ ] **Y-42** [kod] Retrieval hattına reranker aşaması ekle (top-k aday → reranker → LLM) ve şu modelleri karşılaştır: Qwen3-Reranker, Jina Reranker v3, BGE-Reranker-v2-m3, gte-multilingual-reranker-base. Reranker'lı ve reranker'sız nDCG@10 ile gecikme ölçülsün. Bu madde makalede anlatılıp kodda olmayan reranker modülünü de kapatır (Y-12). · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: Uğurhan'ın model listesi (2026-11-07)
- [ ] **Y-44** [kod] LLM katmanına API ile çalışan sağlayıcılar ekle: Gemini ve EVREN modelleri. Anahtar `.env` dosyasından okunsun, repoya girmesin. Bu modeller yerel olmadığı için belge içeriği dışarıya gider, makalede yalnızca karşılaştırma tabanı (üst sınır) olarak ayrı raporlansın, "tamamen yerel" iddiasının içine katılmasın. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: Uğurhan'ın model listesi (2026-11-07)

## Repo

- [ ] **Y-25** [repo] `benchmark.py` içindeki mutlak `BASE_DIR` yolunu kaldır, fine-tune modelleri HF Hub'a yükle, `requirements.txt` sürümlerini sabitle. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: benchmark.py:146
- [ ] **Y-26** [repo] Büyük dosyaları repodan çıkar: kökteki ve `CUSTOM_DATASET/` içindeki iki ayrı `metin_dosyasi.json` (24 MB'lık kopyalar), `DB/` verileri, `.~lock` dosyası. Yerlerine indirme betiği koy. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: repo kökü
- [ ] **Y-27** [repo] `.claude/settings.json` kişisel izinler içeriyor (`git push *`, `sudo dpkg`, kişisel yollar). Repodan çıkarılsın, `.gitignore`'a eklensin. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: .claude/settings.json
- [ ] **Y-28** [repo] En az ingest ve search için `pytest` testleri yaz, lisans dosyası ekle. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: repo kökü
- [ ] **Y-35** [repo] `UI/backend/` (eski backend) ve `DB/*` ilk deney betiklerini `docs/arsiv/` altına taşı ya da sil. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: makale/incelemeler/2026-11-07-ugurhan-claude-kod-2.md#4
- [ ] **Y-40** [repo] Hazır setler için indirme ve hazırlama betiği yaz (`data/indir.py` gibi). Her setin lisansını kendi sayfasından kontrol edip `data/LISANSLAR.md` dosyasına yaz. Veriler repoya commit'lenmesin. · ekleyen: Uğurhan (Claude) · sorumlu: — · kaynak: makale/kararlar.md#K-01
