
# Kod review 2: veri seti üretimi, benchmark iç işleyişi, Flutter, yan betikler

Kod: main `9570cc2`. İlk tur: `makale/incelemeler/2026-11-07-ugurhan-claude.md`.
Bu turda okunanlar: `CUSTOM_DATASET/text_search.py`, `squad_dataset.json` (istatistik betiği çalıştırıldı), `benchmark.py` encode/worker/bellek kısmı, `UI/backend/main.py`, Flutter `lib/` (api_services, chat_screen, chat_bubble), `DB/*`.

## 1. Veri seti (en önemli yeni bulgu)

`squad_dataset.json` gerçek SQuAD değil. `text_search.py` ile `ministral-3:8b` aynı 329 PDF'ten soru üretmiş (belge başına 5 parça, parça başına 5 soru, `RecursiveCharacterTextSplitter` 1200/100). Sayılar (betikle sayıldı): 7.194 soru, 1.059 paragraf, 329 belge.

- **1.405 birebir kopya soru** var: aynı paragraf için aynı soru metni birden fazla kez geçiyor. `process_file` içinde tekilleştirme yok (`text_search.py:156-175`). Kalite örneklemi (1000 soru) bu kopyaları da içeriyor.
- **100 soru metni birden fazla farklı paragrafa bağlı.** Bu sorular toplam 595 örneğe denk geliyor. Örnek: "What is the title of the article?" 74 kez ve 50 farklı paragrafta geçiyor. Bu tür bir soruda tek doğru belge yok. Ama qrels her soru kimliğine yalnız tek paragrafı doğru sayıyor (`benchmark.py load_squad_quality`). Gürültülü etiket her modelin skorunu düşürüyor. Fine-tune modeli ise bu soru–paragraf eşleşmesini ezberleyebiliyor.
- **Soruların yaklaşık %30'u bağlama bağımlı.** "the article", "this study", "the paper" gibi ifadeler içeriyor (kaba anahtar kelime sayımı). Bağlamı görmeden bu soruların hangi paragrafa ait olduğunu bilmek imkânsız. Base modellerin 0,3–0,5 nDCG almasının ikinci açıklaması bu. Birincisi önek eksikliği, üçüncüsü sızıntı.
- Üretilen soruların doğruluğu hiç insan kontrolünden geçmemiş. Cevabın metinde geçip geçmediğine yalnızca substring kontrolüyle bakılıyor. Makalede veri setinin LLM ile üretildiği hiç söylenmiyor.
- Kalite seti 1200 karakterlik parçalardan, hız korpusu 500/50 parçalardan oluşuyor. İkisi aynı bölmeyi kullanmadığı için qrels doğrudan 52 bin chunk'a taşınamaz. Y-04 için ya aynı bölme kullanılmalı ya da paragraf → chunk eşlemesi yapılmalı.

**Öneri:** Soruları normalize ederek tekilleştir. Birden fazla paragrafa uyan ve bağlama bağımlı soruları ele ya da yeniden yazdır. 100–200 soruluk bir alt kümeyi iki kişi elle doğrulasın ve uyum oranını raporla. Makaleye "LLM ile üretilmiş sentetik soru seti" diye açıkça yaz.

## 2. Benchmark iç işleyişi

- **CPU'ya düşme yok.** `encode_with_fallback` docstring'i (`benchmark.py:414`) ve README "son çare CPU" diyor. Kod ise yalnız batch'i yarıya indiriyor, batch=1'de de olmazsa hata fırlatıyor. `qwen_lora`, `bge_m3_*` OOM'ları büyük ihtimalle bu yüzden.
- **"Peak RAM" peak değil.** `_peak_mem` yalnızca işin sonunda `rss` okuyor (`benchmark.py:1055-1060`). Daha önemlisi, ölçülen yalnızca **istemci sürecinin** belleği. Qdrant ve Weaviate Docker'da ayrı süreç, onların sunucu belleği hiç ölçülmüyor. Makaledeki "Qdrant/Weaviate ~440 MB ile en az RAM" karşılaştırması bu yüzden geçersiz. Gömülü DB'lerin (Chroma, LanceDB, Milvus Lite) belleği istemci sürecinin içinde sayılıyor. Gerekenler: `docker stats` ya da cgroup ile sunucu belleği, `psutil` örnekleme thread'i ile gerçek tepe değer.
- **Yazma süreleri aynı işi ölçmüyor (hipotez).** Qdrant HNSW'yi yazmadan sonra arka planda kuruyor, Chroma her `add` çağrısında indeksi senkron kuruyor. "Chroma en yavaş yazıyor, Qdrant hızlı" farkının bir kısmı indeks kurulumunun ölçüme dahil olup olmamasından gelebilir. Her DB'de "yazma + indeks hazır olana kadar" süresi ölçülmeli (Qdrant'ta `wait=True` ve optimizer durumu, Milvus'ta `flush` + `load`).
- Sorgu başına sonuç farklı veri döndürüyor. Milvus `output_fields=["text"]` ile metni getiriyor, Qdrant varsayılan ayarda payload getirmiyor, LanceDB her sorguda DataFrame kuruyor. Bu da hız farkının bir parçası.

## 3. Flutter arayüzü

- **Yükleme hatası "başarılı" görünüyor.** `uploadPdfBase64` (`api_services.dart:81-90`) yalnızca HTTP 200'e bakıyor, `jsonResponse['status']`'ü kontrol etmiyor. Backend hata durumunda da 200 döndüğü için (Y-24) kullanıcı yeşil "başarılı" bildirimi görüyor.
- **Türkçe karakter bozulması.** `normalChat` ve `uploadPdfBase64` gövdeyi `response.body` ile okuyor, `getApiResponse` ise `utf8.decode(bodyBytes)` ile. Sunucu charset göndermezse ilk ikisinde Türkçe karakterler bozuk çıkar.
- **Zaman aşımı yok.** Hiçbir `http.post` çağrısında timeout yok. Ollama takılırsa arayüz sonsuza kadar "analiz ediliyor" durumunda bekler.
- **Konuşma geçmişi backend'e gitmiyor.** Her mesaj tek başına gönderiliyor. Arayüz "sohbet" gibi görünse de takip sorusu ("peki bunun maliyeti?") bağlamsız kalıyor. Makaledeki zamir çözümleme ve sorgu yeniden yazma (ICR) anlatımı uygulamada yok.
- **Yükleme ekranı yanlış mod gösteriyor.** Deep Research kapalıyken "X modunda arama…" yazıyor (`chat_screen.dart:621-623`), oysa normal sohbet hiç arama yapmıyor. Ayrıca normal sohbete `scenario/db/embedding` gönderiliyor ama backend bunları kullanmıyor.
- Model listesinde çeviri modeli `translategemma:4b` var (`chat_screen.dart:67`). Liste elle yazılmış, Ollama'da yüklü modellerle senkron değil (`/api/tags` ile alınabilir).
- PDF'ler multipart yerine base64 JSON ile gönderiliyor: %33 şişme ve büyük dosyada bellek baskısı. Backend'de `/upload/pdf` multipart ucu var ama kullanılmıyor.
- Yüklenen belge listesi (`_uploadedDocs`) kaydedilmiyor, uygulama kapanınca kayboluyor.
- Android `main` manifest'inde `INTERNET` izni yok (yalnız debug/profile'da var), yani release derlemesi ağa çıkamaz. Ayrıca `127.0.0.1` Android'de telefonun kendisini gösterir. Masaüstü hedefleniyorsa bu sorun değil, ama makale "çapraz platform" diyor.

## 4. Yan betikler ve ölü kod

- `UI/backend/main.py` eski bir backend. Farklı şema kullanıyor (`girilen_metin`), göreli yolu `../../yerel_veritabani` ve LLM çağrısı yok. Kafa karıştırıyor, silinmeli ya da `docs/arsiv`'e taşınmalı.
- `DB/pgvector/*.py` içinde düz metin parola var (`password="yeni_sifre"`, `write_vector_database.py:14`, `sorgu.py:12`). Repo public. Karar (Uğurhan, 2026-11-07): parolalar şimdilik kalıyor.
- `DB/*` klasöründekiler ilk deney betikleri. Benchmark bunları kullanmıyor ve makaledeki pgvector bu çalışmada yok. Ya `deneyler/` altına taşınmalı ya da silinmeli.

## 5. Yapılacaklar'a eklenen maddeler

- Y-29 [deney] Soru setini tekilleştir, çok paragraflı ve bağlama bağımlı soruları ele, 100-200 soruyu elle doğrula.
- Y-30 [makale] Veri setinin LLM (ministral-3:8b) ile sentetik üretildiğini ve üretim protokolünü yaz.
- Y-31 [deney] Bellek ölçümünü düzelt: sunucu DB'lerinde `docker stats`, gerçek tepe için örnekleme.
- Y-32 [deney] Yazma süresine indeks hazır olma süresini dahil et, ya da ikisini ayrı ölç.
- Y-33 [kod] OOM durumunda gerçekten CPU'ya düş, ya da docstring ve README'yi düzelt.
- Y-34 [kod] Flutter: yüklemede `status` kontrolü, `utf8.decode`, timeout, konuşma geçmişinin backend'e gönderilmesi.
- Y-35 [repo] `UI/backend/` ve `DB/*` ilk deney betiklerini arşivle.
