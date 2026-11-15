# Kararlar

Ekip olarak verilen kararlar. Her karar tarih, kimin verdiği ve gerekçesiyle buraya yazılır.
Karar değişirse eski kayıt silinmez, altına yeni tarihle güncelleme eklenir.

## K-01 · 2026-11-07 · Veri seti: başlangıçta yalnızca hazır setler (seçenek A)

- **Karar veren:** Uğurhan. Yüksel ve Hakan itiraz edecekse `tartışma` etiketiyle issue açsın.
- **Karar:** Deneyler önce yalnızca hazır ve standart veri setleriyle yapılacak. Kendi topladığımız korpusu temizleyip vaka çalışması olarak eklemek (seçenek C) sonraki aşamaya bırakıldı.
- **Hangi iddia hangi setle ölçülecek:**
  - Embedding modeli kalitesi: BEIR biyomedikal setleri (TREC-COVID, NFCorpus, SciFact). Türkçe istenirse TR-MTEB'in retrieval görevleri eklenir, makine çevirisi olduğu yazılır.
  - Vektör DB hızı, ölçek ve ANN recall: VectorDBBench'in Cohere 1M veri seti (768 boyut). Ground truth exact kNN.
  - Uçtan uca PDF RAG: UDA benchmark'ının akademik PDF kısmı.
- **Gerekçe:** Şu anki soru seti LLM ile üretilmiş sentetik bir set. İçinde kopyalar ve belirsiz sorular var, fine-tune verisiyle de karışıyor. Bkz. `makale/incelemeler/2026-11-07-ugurhan-claude-kod-2.md`. Hazır setler sızıntısız, kıyaslanabilir ve hakemler tarafından kabul görüyor.
- **Sonuç:** Mevcut `squad_dataset.json` ile ilgili temizleme maddeleri (Y-01, Y-04, Y-29, Y-30) C aşamasına ertelendi. Yeni maddeler Y-36..Y-40.
- **Seçenekler ve kaynaklar:** A, B ve C'nin karşılaştırması aşağıda.

### Seçeneklerin özeti

| Seçenek | İçerik | Artı | Eksi |
| --- | --- | --- | --- |
| A (seçildi) | Yalnızca hazır setler | Hakem kabulü, kıyaslanabilir, sızıntı ve lisans sorunu yok | "Kurumsal kapalı ağ" hikâyesi zayıflar, Türkçe yalnızca çeviri |
| B | Yalnızca kendi korpusumuz | Hikâyeyle uyumlu | 200-500 soruyu elle etiketlemek gerekir, kıyaslanamaz, lisans işi büyük |
| C (sonra) | A + kendi korpusumuz (PMC OA CC BY PDF'leri, filtrelenmiş sentetik sorular, 200 sorunun elle doğrulanması) | A'nın kanıtı ve projenin özgün tarafı birlikte | En fazla iş |

Kaynaklar:
- BEIR: https://github.com/beir-cellar/beir/wiki/Datasets-available
- VectorDBBench: https://github.com/zilliztech/VectorDBBench
- UDA (NeurIPS 2024 D&B): https://github.com/qinchuanhui/UDA-Benchmark
- TR-MTEB: https://aclanthology.org/2025.findings-emnlp.471/
- pdfQA (alternatif): https://arxiv.org/abs/2601.02285
- PMC OA (C aşaması için): https://catalog.data.gov/dataset/pubmed-central-open-access-subset-pmc-oa

## K-02 · 2026-11-07 · Ortak veri tarifi ve setlerin rolleri

- **Karar veren:** Yüksel (taslak ve denetim: Claude, 4 alt agent). Uğurhan ve Hakan itiraz edecekse `tartışma` etiketiyle issue açsın.
- **Karar:** Üç kişi aynı veriyi, aynı bölmeyi ve aynı ölçüm kuralını kullanır. Tarif `data/VERI-TARIFI.md` dosyasındadır, sürümlüdür, değişirse eski sürüm silinmez. Her sonuç tarif sürümünü yazar.
- **Roller (tarif sürüm 1):**
  - SciFact: fine-tune train + test, sızıntı temizliğiyle (test pozitiflerinin %64'ü train'de de pozitif).
  - NFCorpus: yalnız zero-shot test (test pozitiflerinin %99,6'sı train'de var).
  - TREC-COVID: yalnız held-out test.
  - SciFact-TR ve NFCorpus-TR: yalnız Türkçe test, makine çevirisi, temizlikle, kendi betiğimizle.
  - Cohere 1M: yalnız DB hız/ölçek/ANN recall (metin yok, kendi modellerimizle uyumsuz).
  - UDA PaperTab + PaperText: yalnız uçtan uca PDF RAG testi (train split'i yok).
- **K-01'e etkisi:** MsMarco-TR kapsamdan çıkarıldı (Y-39 yalnız Scifact-TR ve NFCorpus-TR ile uygulanır): test sorgularının 4.421'i train'de de var, BEIR MS MARCO değil, lisansı kısıtlı. K-01'in diğer kararları geçerli.
- **Gerekçe:** Kanıtlar `data/VERI-TARIFI.md` içinde, her biri ölçüldü / kaynaktan okundu / doğrulanamadı etiketiyle. Denetim alt agent'larla yapıldı, bulgular betikle yeniden üretilene kadar makalede "kanıtlandı" diye yazılmaz.
- **Açık:** SciFact train küçük (809 sorgu). Daha büyük fine-tune verisi gerekirse yeni karar. Lisans doğrulamaları Y-56.
- **Yeni maddeler:** Y-54..Y-58.

## K-03 · 2026-11-11 · UDA ilk turda yalnız token F1, LLM-hakem sonraya (Y-57)

- **Karar veren:** Yüksel. Uğurhan ve Hakan itiraz edecekse `tartışma` etiketiyle issue açsın.
- **Karar:** UDA PaperTab ve PaperText ilk turda ~300 soruluk sabit alt kümeyle (seed 42, orantılı: 37 PaperTab, 263 PaperText) ve yalnız Qasper tarzı token F1 ile ölçülür. LLM-hakem şimdilik eklenmez.
- **Gerekçe:** Sonuçlar Qasper'la karşılaştırılabilir kalır; hakem maliyet ve tutarsızlık getirir. Cevapların ~%71'i serbest cümle olduğu için token F1 kaba kalır, bu sınır makalede yazılır.
- **Yeniden açma koşulu:** İlk tur sonuçları anlamsız derecede düşük çıkarsa hakem ayrı bir kararla eklenir.
- **Dosyalar:** `data/uda_altkume.py`, `data/uda_f1.py`, `data/listeler/uda_altkume_v1.json`.
