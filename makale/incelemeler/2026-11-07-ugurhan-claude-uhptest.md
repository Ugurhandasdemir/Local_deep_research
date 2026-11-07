
# uhptest koşusu, 2026-11-07

Kim: Uğurhan (Claude), Yüksel'in `uhptest` skill'i ile (github.com/Yukseltt/uhptest). Kod: main `8f964c8`. Ortak vault'ta da var: `knowledge/concepts/uhptest-2026-11-07.md`.
Testler repoya yazılmadı. Skill'in paylaşılan repo kuralına uyuldu: kaynak dosyaların kopyası scratchpad'e alındı, ChromaDB, LangChain ve sentence-transformers sahte modülle değiştirildi. Test edilen davranış gerçek kod. Repo `git status` temiz.
Hedefler: `vector_indexes.py`, `main.py` (FastAPI uçları), `benchmark.py` (`compute_quality_metrics`, `load_squad_quality`), `CUSTOM_DATASET/text_search.py`.
Sonuç: 40 test fonksiyonu (parametrelerle 56 durum), 27 geçti, 29 başarısız, 6 açık sözleşme sorusu.

## Bulgular (önem sırasıyla)

1. **Taranmış PDF "başarılı" olarak indeksleniyor.** Metni olmayan 10 sayfalık PDF'te `extract_text` boş dönüyor, ama kod her sayfaya `---PAGE n---` ekliyor. Bu yüzden metin boş görünmüyor. Sonuç: `status: success`, `chunks_added: 1` ve yalnız sayfa işaretlerinden oluşan bir chunk indekse giriyor. (`test_upload_base64_scanned_pdf_without_text_is_reported_as_no_text`, `test_chunk_text_only_page_markers_should_not_produce_chunks`)
2. **Bir modelin yüklenememesi tüm indekslemeyi durduruyor.** `index_document_for_all_scenarios` içinde `_encode` try bloğunun dışında. `models/` repoda olmadığı için temiz bir klonda her yükleme hata veriyor, oysa varsayılan Chroma'ya yazma zaten yapılmış oluyor (yarım kalmış durum). (`test_index_all_scenarios_encode_failure_is_reported_per_target`)
3. **Metrikler 1'i aşabiliyor.** Aynı belge kimliği listede tekrar ederse recall@10 = 3,0 ve nDCG@10 = 2,13 çıkıyor. (`test_metrics_stay_within_unit_interval_when_retrieval_has_duplicates`)
4. **Soru setinde etiket gürültüsü.** Aynı metinli soru farklı paragraflarda geçtiğinde her soru kimliği yalnız kendi paragrafını doğru sayıyor. Aynı paragraftaki kopya sorular da ayrı sorgu olarak sayılıyor. Gerçek veride 1.405 kopya soru ve 100 çok paragraflı soru metni var. (`test_squad_*`)
5. **Türkçe İ, cevap aralığını kaydırıyor.** `find_answer_start` büyük-küçük harf duyarsız aramada `lower()` kullanıyor. "İ" küçültülünce iki kod noktası oluyor ve bulunan indeks bir karakter kayıyor: "İzmir ve Ankara" metninde "ankara" araması `'nkara '` döndürüyor. "İLK" gibi bir ifadede ise cevap hiç bulunamıyor ve soru sessizce atılıyor. Bugünkü veri İngilizce, Türkçe PDF'lerde sorun çıkar. (`test_find_answer_start_case_insensitive_span_matches_answer`)
6. **LLM çıktısı ayrıştırmada sessiz kayıp ve çökme.** Açgözlü regex, metinde iki köşeli parantez grubu varsa `[]` döndürüyor ve soru-cevaplar kayboluyor. Dict olmayan bir liste dönerse `process_file`'daki `qa.get` tüm üretimi çökertiyor. (`test_extract_json_*`)
7. **Kullanıcıya yanlış mesaj.** Üç indeksin üçü de hata verse bile mesaj "başarıyla işlendi (0/3 ...)" oluyor. Belge hiç chunk üretmediğinde de "başarıyla işlendi" diyor. (`test_processed_message_*`)
8. **Geçersiz dosya kalıcı kalıyor.** PDF olmayan içerik ya da bozuk base64 (geçersiz karakterler doğrulama yapılmadan atlanıyor) önce diske ve `pdfs.db`'ye yazılıyor, sonra hata dönüyor. (`test_upload_base64_non_pdf_content_is_not_persisted`, `..._invalid_characters_...`)
9. **Senaryo adındaki yazım hatası sessizce yutuluyor.** `Fast`, `FAST`, `fastt` gibi değerler hata vermeden `balanced` (Weaviate) senaryosuna düşüyor. (`test_resolve_config_unknown_or_miscased_scenario_is_rejected`)
10. **Küçük sorunlar.** Desteklenmeyen DB seçiminde HTTP 200 dönüyor (Y-24). Aramada metin alanı `None` gelirse istek çöküyor. Windows yolu (`C:\...\rapor.pdf`) Linux'ta olduğu gibi dosya adı oluyor.

## Açık sözleşme soruları (karar kişilerin)

- 50 karakter ve altındaki belge hiç indekslenmesin mi? Şu an sessizce atılıyor.
- Aynı adla yapılan ikinci yükleme ilkini ezsin mi?
- `.`, `..` ve `.pdf` gibi dosya adları kabul edilsin mi?
- Aynı dosya adının NFC ve NFD yazımı (macOS) aynı belge mi sayılsın?
- qrels'i boş olan sorgu metrik ortalamasına 0 olarak girsin mi?
- Metni aynı olan soruların hepsinin paragrafları doğru sayılsın mı?

`UHPTEST: BLOCK written=56 passed=27 failed=29 open_questions=6`

## Yapılacaklara eklenen maddeler

Bulgu 1 → Y-45, 2 → Y-46, 3 → Y-47, 4-6 → Y-48 (C aşamasında Y-29 ile birlikte), 7 → Y-49, 8 → Y-50, 9 → Y-51, 10 → Y-52 (HTTP kodu için Y-24), açık sorular → Y-53.
