
# Local_deep_research: kod review + makale eksikleri

Kaynaklar: repo `Ugurhandasdemir/Local_deep_research` (HEAD 455524d), `Local Deep Research_makale.pdf` (tamamı okundu), `multi_model_benchmark_results.json/.md`, `train_models/*.ipynb`, `main.py`, `vector_indexes.py`, `benchmark.py`.

## A. En kritik 5 sorun (makale bunlarla reddedilir)

1. **Train/test sızıntısı (veri sızıntısı).** `bge_squad`, `all_mini_l6`, `snowflake_arctic_l`, `bge_m3_fine` modelleri `squad_dataset.json` ile fine-tune edildi. Notebook'larda train/val ayrımı yok (`eval_df = train_df.sample(...)`). Kalite benchmark'ı (`load_squad_quality`) aynı dosyadan 1000 soru örnekliyor. Yani "en iyi model bge_squad, nDCG@10=0,9008" bulgusu ezberin ölçümü. Kanıt: aynı mimarinin base hali `bge_squad_base` 0,4876, `all_mini_l6_base` 0,4195; fine-tune hali 0,9008 / 0,8188. Makalenin ana iddiası (özet, sonuç, Tablo 2-4) bu yüzden geçersiz. Çözüm: soruları/paragrafları baştan train/dev/test ayır (paragraf bazlı, aynı bağlam iki tarafa düşmesin), sonuçları yalnız held-out testte raporla; ayrıca BEIR/MS MARCO/SciFact gibi harici bir sette kontrol et.
2. **"52.331 kayıt üzerinde doğruluk" yanlış.** Kalite aşamasının korpusu 1.059 SQuAD paragrafı (`corpus_count: 1059`), 52.331 yalnız yazma/arama hızı korpusu. Makale IV.B "52.331 vektör kaydı üzerinde ... doğruluk" diyor. ANN davranışı 1.059 vektörde ile 52 bin vektörde çok farklı; recall-hız dengesi anlamlı değil. Düzelt ya da kaliteyi büyük korpusta (örn. 52k chunk içine gömülmüş SQuAD paragrafları) yeniden ölç.
3. **Tüm DB'lerde aynı nDCG (0,9008).** Bu "DB doğruluğu eşit" demek değil, ANN recall'ünü hiç ölçmediğini gösteriyor: 1.059 vektörde HNSW fiilen brute-force. Makale "doğruluk-hız dengesi" (Şekil 4, 11) sunuyor ama exact-kNN'e göre ANN recall@k (ör. recall of ground-truth neighbors) yok. HNSW parametreleri (M, ef_construction) da DB'ler arasında eşitlenmedi; her DB varsayılanda.
4. **Hız ölçümü adil değil.** `HNSW_batch` (1000 sorguyu tek çağrıda) ile tek tek döngüde çağrılan DB'ler (LanceDB `to_pandas()` her sorguda, Weaviate gRPC) aynı tabloda sıralanıyor. ChromaDB 3384 QPS, LanceDB 2,2 QPS karşılaştırması yapıdan çıkan fark; DB farkı değil. Ayrıca: Qdrant/Weaviate Docker sunucu (ağ + serileştirme), Milvus Lite/Chroma/LanceDB gömülü süreç içi; bunlar aynı kategoride kıyaslanmış. "Milvus dağıtık mimarisi" yorumu yanlış: kullanılan şey Milvus **Lite** (tek dosya, dağıtık değil). 
5. **Rapor edilen sayılar kendi çıktısıyla çelişiyor / yeniden üretilemiyor.** Repodaki `multi_model_benchmark_results.json`'da `search` bölümünün 25/25 girdisi `ModuleNotFoundError: chromadb` hatalı, `metadata.query_count: 10` (yani fallback sorgu listesi), `.md` "TEST_RUNS=10" diyor, kod `TEST_RUNS=3`, makale "2 warmup + 5 tur" diyor. Üç farklı protokol. README "26 model", makale "28 model", JSON'da 28 var ama `qwen_lora`/`bge_m3_*` OOM yazıyor (JSON'da yine de değerli). Makale Tablo 1 metin "5501,4" vs tablo "5510,4"; Milvus `bge_squad` yazma 10,25 s (makale) vs 23,6 s (README); Tablo 4'te LanceDB QPS "2.Oca" (Excel tarih hatası, "2,1" olacaktı). Hepsi tek bir temiz koşudan otomatik üretilmeli.

## B. Makalede eksikler / hatalar

**İçerik–iddia uyumsuzluğu**
- Özet ve giriş "iki aşamalı, netleştirici soru soran derin araştırma asistanı" vaat ediyor (Şekil 1: sorgu üret → ara → özetle → tekrar düşün döngüsü; III.C sorgu netleştirme; III bölümünde hibrit + reranker modülü). **Kodda bunların hiçbiri yok.** `/ask/question/ai` tek seferlik: encode → top-5 → tek prompt → cevap. Döngü yok, netleştirme sorusu yok, reranker (cross-encoder) yok, hibrit arama üretimde yok (yalnız benchmark'ta Weaviate), HyDE/RAG-Fusion yok, "çoklu embedding modelini birleştirme" yok (kullanıcı birini seçiyor). Ya özelliği yaz ya da makaleyi gerçeğe indir ("tek geçişli RAG").
- "Dinamik GPU/CPU yönetimi: model idle CPU'da, gerektiğinde GPU'ya" (III.B). `vector_indexes._embedding_model` modeli **her zaman `device="cpu"`** yüklüyor, GPU geçişi yok. Yalnız `benchmark.py` OOM→CPU fallback'i var. İddia kodda doğrulanmıyor.
- "Halüsinasyondan arındırılmış" (IV, Sonuç): hiçbir üretim/halüsinasyon/groundedness ölçümü yok. RAG üçlüsü (context relevance, groundedness, answer relevance) literatür bölümünde anlatılıp **hiç ölçülmemiş**. LLM cevap kalitesi (nemotron/ministral/granite/medgemma) için tek bir deney yok; "tamamen yerel RAG" başlığına rağmen LLM tarafı değerlendirilmemiş.
- "Sistem ... güvenliği kanıtlamıştır" (Sonuç): güvenlik/gizlilik için tehdit modeli, ölçüm, test yok. Kod CORS `*` + `allow_credentials=True`, `host=0.0.0.0`, kimlik doğrulama yok, rate limit yok: "kurum içi güvenli" iddiasıyla çelişiyor.
- Kullanıcı çalışması yok (Flutter arayüz "verimliliği artırır" deniyor; SUS/zaman ölçümü yok). "Verimlilik artışı vaat etmektedir" kanıtsız.
- Dil: veri seti İngilizce (healthcare-AI SQuAD, bge-large-en), cevap Türkçe prompt, gerekçe Türkçe kurum. Türkçe retrieval hiç ölçülmemiş; 'multilingual' modeller İngilizce sette değerlendirilmiş. Başlık/özet İngilizce, gövde Türkçe, bölüm başlıkları karışık; IEEE şablonunda tek dil olmalı.

**Deney tasarımı**
- Veri seti tanımı yok: 329 PDF, SQuAD formatındaki soruların kim/nasıl üretildiği (LLM mi, elle mi), kaç soru, paragraf sayısı, lisans yok. `metin_dosyasi.json` (24 MB) ile `squad_dataset.json` ilişkisi belirsiz (yazma korpusu 52k chunk, kalite korpusu 1.059 paragraf; farklı).
- İstatistiksel anlamlılık yok: tek koşu, tek seed, güven aralığı/std yok, 28×5 karşılaştırmada çoklu test düzeltmesi yok. QPS için 3-5 tur, p99 aslında "max proxy" (kendi .md'sinde yazıyor). Gerçek p99 için en az yüzlerce tekil sorgu gecikmesi gerekir; mevcut ölçüm toplu çağrının duvar süresi.
- Donanım/versiyon bilgisi eksik: CPU, RAM, GPU modeli, disk, OS, DB sürümleri, Docker kaynak limiti, chunk parametreleri. Soğuk/sıcak cache ayrımı yok. Yazma testi `batch=32` ve DB'lerin farklı API'leri (Weaviate `dynamic batch`, Chroma `add`, Milvus `insert`) ile; "yazma süresi" indeks kurulumu dahil mi, DB'ye göre değişiyor (Milvus Lite'ta HNSW build zamanı, Qdrant'ta arka plan indeksleme). Adil olmayan "yazma hızı" yorumu.
- Embedding boyutu–yazma süresi korelasyonu (Şekil 9-10): boyut ile model ailesi karıştı (384=MiniLM/e5-small, 1024=BGE/Arctic); nedensel yorum ("Milvus dağıtık indeksleme avantajı") kanıtsız. Pearson p-değerleri/n verilmemiş.
- Fine-tune süreci anlatılmamış (veri, epoch, kayıp, hard-negative madenciliği, donanım). Makalede neredeyse tek satır "bge_squad". Base vs fine-tune farkı en güçlü sonucunuz olabilirdi, ama sızıntı yüzünden şu an savunulamaz (A.1).
- Kıyas yok: BM25 baseline'ı kaliteyle ölçülmemiş (Weaviate BM25/hibrit hız ölçüldü, nDCG yok); "semantik arama keyword'den iyidir" ana motivasyonu hiç sayıyla gösterilmemiş. Hibrit alpha taraması sonuçları makalede yok.
- İlgili çalışmalar: ANN-Benchmarks, VectorDBBench, VDBBench, BEIR, MTEB dışında vektör DB karşılaştırma makaleleri yok; katkı iddiası ("literatüre kritik katkı") konumlanmamış.

**Yazım / atıf hataları (doğrudan reddedilme sebebi olur)**
- Atıf numaraları kaymış/yanlış: metinde [20] HNSW diye geçiyor, listede [20]=RAGAS, HNSW=[21]; metinde [22]=ANN-Benchmarks fakat Şekil 4 Qdrant sonucuna atıf; [25]=BGE denirken listede [25] **nötrino salınımı kuantum kaynak teorisi** makalesi (alakasız); [23]=NEFTune "Weaviate hibrit arama" için; [24]=Aggarwal 2001 "p99 gecikme" için; [27] BGE-M3 "GPU'ya çekip bellek temizleme" için; [15] ve [16] listede **aynı kayıt (PagedAttention) iki kez**; [11] "nDCG" için ColBERT/Trappolini; [12] RAG üçlüsü için ColBERT; [13] TRUE faktüel tutarlılık "latency" için. Atıf listesi bu metinle eşleştirilip baştan kurulmalı. Ayrıca metindeki "[26]" (Dean & Barroso) kullanılmamış, [28] MTEB cümleyle uyumsuz.
- Hiç alıntı yapılmayanlar: Milvus, Qdrant, Chroma, Weaviate, LanceDB, pgvector (repoda `DB/pgvector` var ama makalede yok) için kaynak/belge sürümü yok; BEIR nDCG tanımı; SQuAD (Rajpurkar 2016) kendisi atıflanmamış.
- Yazım: "Chorame/ChoramaDB/ChormaDB", "embbeding", "maaliyet", "Şekil 11 başlığı hem Şekil 11 hem 11'de 12 yazısı", Tablo 2'de Qdrant/Weaviate satırlarında bozuk hücre ("chromad b"), "Tablo 1" metin 5501,4 vs 5510,4, "3384.1 QPS" ile "p99 3,083 ms" aynı satırda tutarsız (1000 sorguluk toplu çağrı 0,3 s sürerken p99=3 ms?). Başlıklar "INTRODUCTİON / LİTERATURE REVİEW / MATERİAL AND METHOD / DİSCUSSİON" (Türkçe İ'li). Bölümlerin sırası bozuk (IV'ün ortasında III/Discussion metni, Tablo 3/4 yer değişmiş). Özette "28 embedding" doğru, README 26.
- Özet "closed, internal network ... completely eliminates privacy risks" mutlak ifade. "Kanıtlamıştır/vaat etmektedir" gibi abartılı sonuç dili.
- Yazar bilgileri: ORCID/afiliasyon metni karışmış, "Department of Department" kalıbı.

## C. Repo (kod) eksikleri

**Güvenlik**
- `main.py:18-24` CORS `allow_origins=["*"]` + `allow_credentials=True`, kimlik doğrulama yok, `uvicorn host=0.0.0.0` ve `reload=True` (üretim değil).
- `/upload/pdf*` dosya boyutu/sayfa sınırı yok (RAM/disk DoS), tüm PDF'ler `pdfs.db` BLOB + disk (çift depolama), aynı ad aynı dosyayı sessizce ezer (kullanıcı ayrımı yok → çok kullanıcılı ortamda başkasının belgesi silinir/okunur: `/pdfs/{filename:path}` herkese açık).
- Prompt injection önlemi yok: PDF içeriği doğrudan prompta ("DOSYA ADI/İÇERİK") konuyor; makale "istem enjeksiyonu riskini" gerekçe gösterdiği halde savunma yok.
- `/pdfs/{filename:path}` `safe_pdf_filename` ile `basename` alıyor (iyi), ama `GET`'te dönüş `Response` başlıksız; hata yolunda `str(e)` istemciye sızıyor (iç yol bilgisi).

**Doğruluk / hatalar**
- `main.py` ile `vector_indexes.py` **iki ayrı indeksleme hattı**: yükleme hem `upsert_default_chroma_chunks` (3000 karakter, Chroma varsayılan embedding) hem `index_document_for_all_scenarios` (500/50) çalıştırıyor. `hnsw_vector_search`/`get_collection` ölü kod (kullanılmıyor), `DB_PATH` `"chorame"` yazım hatası.
- `index_document_for_all_scenarios` her yüklemede **3 senaryoyu birden** (Chroma+Weaviate+Milvus) indeksliyor; Weaviate/Docker kapalıysa hata yutulup "1/3 başarılı" mesajı dönüyor ama HTTP 200 "success".
- Weaviate indeksleme: chunk başına `exists` + `insert` (N+1 ağ çağrısı), `batch` kullanılmıyor; bu bizzat makalenin "yazma hızı" iddiasıyla çelişen üretim kodu. Milvus/Weaviate kimlikleri `model:filename:idx` → PDF güncellenince **eski daha uzun sürümün artık chunk'ları silinmiyor** (stale chunk).
- `_stable_int_id`: 64-bit mod çarpışma riski düşük ama `1e12 + (mod 8e18)` int64 taşma sınırı kontrolsüz; README ile uyumlu ama doğrulanmamış.
- `SentenceTransformer` her istekte `lru_cache(4)`; `MilvusClient(db_path)` ve Weaviate bağlantısı **her aramada yeniden açılıyor** (Milvus Lite dosya kilidi, Weaviate client aç/kapa) → gecikme ve kilit çakışması; bağlantı havuzu yok.
- `_search_milvus`: `hit["distance"]` COSINE'da benzerlik, Chroma/Weaviate'te `1-dist`; skorların anlamı DB'ye göre farklı, UI "benzerlik" olarak gösteriyor. Aynı kaynak dosyadan 5 chunk gelebilir; tekilleştirme/MMR yok. Eşik (min benzerlik) yok → alakasız soruda bile en yakın 5 chunk LLM'e gider, "Bilgi bulunamadı" davranışı yalnız prompta bırakılmış.
- Bağlam penceresi: `num_ctx=2048` iken 5 chunk×500 char + prompt + `num_predict=512` sınırında; sessiz kırpma olabilir (token sayımı yok). Türkçe PDF'te chunk 500 *karakter* (≈ 125 token) çok küçük; chunk ayrımı karakter bazlı, cümle/paragraf/sayfa farkında değil; sayfa numarası metadata'sı yok (makale "hangi sayfadan" diyor, kodda `---PAGE n---` metne gömülüp chunk sınırında kayboluyor, kaynak olarak yalnız dosya adı döner).
- Sorunun kendisi hiç yeniden yazılmıyor/netleştirilmiyor; konuşma geçmişi LLM'e gitmiyor (her istek bağımsız), yani "sohbet" bağlamsız.
- `/normal/chat` aynı `ollama()` ile `temperature=0.7`, `GirdiVerisi` yeniden kullanılıyor; akış (streaming) yok, ama makale "akış tabanlı yanıt gösterimi" diyor.
- Hata yönetimi: her uçta geniş `except Exception` + 200 OK ile `{"status":"error"}`; HTTP durum kodları yok.
- `requirements.txt` sürüm sabitlemesiz (`fastapi`, `uvicorn`, `langchain` …), `pgvector`/`psycopg2` makalede yok ama bağımlılıkta; lock dosyası yok → makale sonuçları yeniden üretilemez.

**Benchmark kodu (`benchmark.py`)**
- `BASE_DIR = "/home/ugo/Documents/Python/bitirememe projesi"` mutlak yol sabit; başka makinede çalışmaz. Docstring'de `write_vector_database.py` eski dosya adı. `models/` (fine-tune ağırlıkları) repoda yok ⇒ `bge_squad`, `all_mini_l6` ve diğer yerel modeller **kimse yeniden üretemez**, HuggingFace'e koyulmamış.
- `metin_dosyasi.json` repoda **iki kopya** (kök + `CUSTOM_DATASET`, 24 MB×2), `DB/` altında LanceDB/Qdrant verisi (~50 MB) ve `.lance` dosyaları, `.~lock.*xlsx#`, `.claude/settings.json` commit'li. Lisans dosyası yok; PDF'lerin kaynağı/telif durumu belirtilmemiş (329 PDF'den korpus dağıtımı hukuken riskli, kaynak listesi yok).
- Doğrulama yok: `measure_time` hataları sessizce `{"error"}` döndürüp sonuçtan düşürüyor (committed JSON'da bütün arama sonuçları hata). Test yok (`tests/` yok), CI yok, tek `README`.
- `p99_proxy` = max (≤10 örnek); adlandırma dürüst ama makale "P99" diye veriyor. Gecikme dağılımı için tekil sorgu zamanları tutulmuyor.
- Qdrant: `exact=True` ölçüldü ama recall kıyası için kullanılmıyor (ANN recall hesabı tam burada yapılabilirdi).
- Sorgu vektörleri önceden encode edilip cache'leniyor ⇒ uçtan uca gecikme (sorgu encode + arama + LLM) hiç ölçülmüyor; kullanıcı açısından gerçek gecikme bilinmiyor. TTFT/TPOT literatürde sayılıp deneyde yok.
- e5 modellerinde `query:`/`passage:` önekleri, BGE'de sorgu talimatı (`Represent this sentence...`) kullanılmıyor ⇒ base model skorları (0,3-0,5) haksız şekilde düşük; "fine-tune çok iyi" izlenimi kısmen ölçüm artefaktı. `max_seq_len` ve normalizasyon (`multi-qa-mpnet-dot` normalize edilmiş cosine ile) modele göre tutarsız.
- UI: Flutter tarafı için test yok, `apiBaseUrl` sabit; `README` ekran görüntüleri tarih 2026-06-18.

## D. Düzeltme yol haritası (öncelik sırasıyla)

1. **Bölme + yeniden ölç.** Paragraf bazlı train/dev/test; yalnız test raporla. Harici set (SciFact/NFCorpus, ya da MS MARCO alt kümesi) ekle. Base modellerde doğru prefix/instruction kullan. (A.1, C benchmark)
2. **Büyük korpusta kalite + ANN recall.** 52k chunk içine SQuAD paragraflarını göm, exact-kNN'e göre ANN recall@10 ölç; her DB'de HNSW M/ef_construction eşitle. (A.2, A.3)
3. **Adil hız protokolü.** Tek-sorgu ve batch'i ayrı tablolarda ver; gömülü vs sunucu DB'leri ayır; 1000 tekil sorgu gecikmesinden gerçek p50/p95/p99; 3+ bağımsız koşu, ortalama±std; donanım/sürüm tablosu. (A.4, A.5)
4. **Uçtan uca RAG değerlendirmesi.** 100-200 soruluk altın set, RAGAS/elle: faithfulness, answer relevance, context precision; 3-4 yerel LLM karşılaştırması; BM25 ve hibrit baseline'larla nDCG. Bu olmadan "derin araştırma asistanı" katkısı yok.
5. **Makale kapsamını kodla eşitle.** Ya netleştirme sorusu + iteratif döngü + reranker + hibrit üretimde uygula, ya da makaleden çıkar. GPU/CPU swap iddiasını ya uygula ya sil.
6. **Atıf listesini baştan kur** (A/B'deki kaymalar, nötrino makalesi, çift PagedAttention), yazım/başlık düzeltmeleri, tek dil.
7. **Repo hijyeni:** mutlak yolları kaldır, modelleri HF Hub'a koy, `requirements` sabitle, büyük JSON/DB'leri `.gitignore` + indirme betiği, `ruff`/`pytest` ile en az ingest/search testleri, CORS/auth/boyut limiti, bağlantı yeniden kullanımı, Weaviate batch.

## Güven notu

Kod ve JSON alıntıları doğrudan dosyalardan; "sızıntı" sonucu notebook'larda ayrı bölme olmaması + aynı `squad_dataset.json` kullanımı + base/fine-tune uçurumundan çıkarım (çalıştırarak doğrulanmadı). Atıf kayması PDF metninden okundu; PDF çıktısı iki sütunlu olduğu için birkaç atıf numarası (özellikle [15]/[16]) görsel olarak teyit edilmeli.

## E. İnternet araştırması: bulguların literatürle desteği (2026-11-07)

| Bulgu | Dış kaynak ne diyor | Kaynak |
| --- | --- | --- |
| ANN recall ölçülmemiş (A.3) | ANN-Benchmarks recall'ı kesin sonuçlara göre ölçer ve "recall'a karşı QPS" eğrisi çizer, parametreleri tarar. Tek noktalık "aynı nDCG" bunun yerine geçmez. Pratik tanım: `len(ann_top_k ∩ exact_top_k)/k`. | [ANN-Benchmarks eleştirisi (YDB)](https://blog.ydb.tech/are-published-ann-benchmarks-dbms-results-trustworthy-f2573eca4e07), [recall kalitesi](https://futureagi.com/blog/evaluating-vector-database-recall-quality-2026/) |
| Batch vs tekil QPS adil değil (A.4) | Batch mod sonuçları bozar; eşzamanlılıkta basit QPS formülü yanlış çıkar; gömülü kütüphaneler yalnız ID döndürür, sunucu DB'ler payload da döndürür. | [YDB](https://blog.ydb.tech/are-published-ann-benchmarks-dbms-results-trustworthy-f2573eca4e07), [Weaviate ANN notları](https://docs.weaviate.io/weaviate/benchmarks/ann) |
| "p99" aslında max (A.5) | Seri istek atan ölçüm service time ölçer, response time değil; coordinated omission p99'u düşük gösterir. Hedef gönderim zamanından ölç, histogram ve p50/p90/p99 ver. Birincil kaynak: Gil Tene "How NOT to Measure Latency", HdrHistogram. | [ScyllaDB](https://www.scylladb.com/2021/04/22/on-coordinated-omission/), [NearForm](https://nearform.com/blog/better-benchmarking-benchmark-tool-lying) |
| Base modeller haksız düşük (C benchmark) | BGE sorgu tarafına "Represent this sentence for searching relevant passages:" ister (pasaja gerekmez); E5 hem sorguya `query: ` hem pasaja `passage: ` ister. Benchmark'ta ikisi de yok. | [bge-large-en-v1.5 kartı](https://huggingface.co/BAAI/bge-large-en-v1.5/discussions/11), [e5-base-v2 kartı](https://huggingface.co/intfloat/e5-base-v2/discussions/2) |
| Üretim kalitesi hiç ölçülmemiş (B) | RAGAS: retrieval için context precision/recall, üretim için faithfulness/answer relevance; çoğu LLM-judge ile. Yerel/gizlilik iddiasında hakem LLM de yerel olmalı, yoksa özel belgeler dışarı gider. En az 50-100 temsilî soru önerilir. | [ragR (arXiv 2604.23515)](https://arxiv.org/pdf/2604.23515), [RAG Evaluation 2026](https://blog.premai.io/rag-evaluation-metrics-frameworks-testing-2026) |
| Vendor/sentetik benchmark gerçekçi değil | Çoğu sayfa 1M sentetik vektör, filtresiz, el ayarlı HNSW ile Recall@10≈0,97 raporlar; kendi vektör ve filtrenizle ölçün. Sizin 1.059'luk kalite korpusu bu yüzden zayıf. | [MariaDB](https://mariadb.com/resources/blog/how-fast-is-mariadb-vector/), [Percona](https://percona.com/blog/benchmarking-vector-indexes) |

**Eklenecek ilgili çalışmalar (makalede yok, konumlandırma için):** ANN-Benchmarks (Aumüller 2020, zaten [22]'de ama yanlış yerde atıflı), VectorDBBench (Zilliz), BEIR (Thakur 2021), MTEB ([28]'de var), RAGAS ([20]'de var ama kullanılmadı), vektör DB anketi ([arXiv 2310.11703](https://arxiv.org/pdf/2310.11703)). Not: aramalar tek bir harness'ta Milvus/Qdrant/Chroma/Weaviate/LanceDB'yi karşılaştıran hakemli bir çalışma döndürmedi; bu, katkı boşluğunuz olabilir ama bunu iddia etmeden önce Google Scholar/arXiv'de ayrıca taranmalı. İnternet kaynakları çoğunlukla blog/vendor; makalede birincil kaynağa (Tene konuşması, model kartları, ANN-Benchmarks makalesi) atıf yapın.
