# Veri tarifi (sürüm 1, 2026-11-07)

Üç kişi farklı modelleri eğitip test edeceği için herkes **aynı veriyi, aynı bölmeyi ve aynı ölçüm kuralını** kullanır.
Karar: `makale/kararlar.md` K-02. Bu dosya o kararın uygulama tarifidir.

- Tarif değişirse eski sürüm silinmez. Aşağıya yeni sürüm ve gerekçe eklenir. Her sonuç hangi tarif sürümüyle alındığını yazar.
- Veriler repoya ve vault'a commit'lenmez. Yalnızca indirme komutu ve sağlama değeri buradadır.
- İndirme: `python3 data/indir.py --liste` (betik bu dosyadaki sürüm, revision ve sağlama değerlerini kullanır; birini değiştirirsen ikisini birlikte güncelle). Lisanslar: `data/LISANSLAR.md`.
- Aşağıdaki sayılar ve bulgular 2026-11-07 tarihli bir denetim koşusundan gelir (alt agent'lar, Claude). Her satırda
  **[ö]** ölçüldü, **[k]** kaynaktan okundu, **[d]** doğrulanamadı. Betikle yeniden üretilene kadar makaleye "kanıtlandı" diye yazılmaz.

## 1. Roller

| Set | Rol | Neden |
| --- | --- | --- |
| BEIR **SciFact** | fine-tune **train** + **test** | Tek temiz train kaynağı. Sızıntı temizliğiyle (bölüm 3) kullanılır. |
| BEIR **NFCorpus** | yalnız **test** (zero-shot) | Test pozitif dokümanlarının %99,6'sı (3115/3128) train pozitifleri arasında [ö]. Fine-tune'a girmez. |
| BEIR **TREC-COVID** | yalnız **test** (held-out) | Yalnız test qrels'i var (50 sorgu) [ö]. Tek başına karar için az. |
| **SciFact-TR**, **NFCorpus-TR** | yalnız **test** (Türkçe, makine çevirisi) | Bölüm 4'teki temizlikle. Fork protokolü kullanılmaz. |
| **MsMarco-TR** | **kullanılmaz** | Test'in bir kısmı train'de var, BEIR MS MARCO değil, lisans kısıtlı (bölüm 4). |
| **VectorDBBench Cohere 1M** | yalnız **DB hız/ölçek/ANN recall** | Metin yok, vektörler Cohere modelinden. Embedding kalitesi için kullanılmaz. |
| **UDA PaperTab + PaperText** | yalnız **uçtan uca PDF RAG testi** | Yalnız `test` split'i var. Fine-tune'a girmez. |
| `CUSTOM_DATASET/` (SQuAD) | bu aşamada kullanılmaz | K-01: C aşamasına ertelendi. |

## 2. Ortak ölçüm kuralları

1. Tek ortak değerlendirme betiği kullanılır (Y-55). Kimse kendi metrik koduyla sonuç raporlamaz.
2. Seed: `42`. Farklı seed denenirse yanına yazılır, 42 sonucu her zaman raporlanır.
3. Retrieval metrikleri: **nDCG@10** (ana), **Recall@100**. Sorgu/doküman önekleri model kartındaki gibi (E5: `query: `/`passage: `, Qwen3: sorgu talimatı).
4. Doküman en çok **512 token** (kırpma). TREC-COVID'de en uzun doküman 18.010 kelime [ö], kırpmasız çalışmaz.
5. Test bölümüne fine-tune sırasında, ayar seçerken veya erken durdurmada dokunulmaz. Seçim için dev (NFCorpus) ya da train'den ayrılmış kısım kullanılır.
6. Her sonuç satırı şunları taşır: tarif sürümü, set + bölüm, model + sürüm/revision, seed, donanım, kütüphane sürümleri, tur sayısı.
7. Sonuçlar betikle üretilir, elle yazılmaz (AI-PROTOKOL §5).

## 3. BEIR (SciFact, NFCorpus, TREC-COVID)

```bash
B=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets
for d in trec-covid nfcorpus scifact; do mkdir -p data/raw; curl -sL -o data/raw/$d.zip $B/$d.zip; unzip -q data/raw/$d.zip -d data/raw/; done
# MD5:    trec-covid ce62140cb23feb9becf6270d0d1fe6d1 | nfcorpus a89dba18a62ef92f7d323ec890a0d38d | scifact 5f7d1de60b170fc8027bb7898e2efca1
# SHA256: trec-covid 120f42a7864d2214234537733c0d2c6684e42fdfafff2c5eacf98afca6656aa0
#         nfcorpus   efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b
#         scifact    536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165
pip install beir==2.2.0
```

HF alternatifi (revision sabit): `mteb/scifact` `cf10ab6856b15b0e670ef8ae5dae4e266c12d035`,
`mteb/nfcorpus` `52ac3f19d3449632d9f00aab0ad34a110fc03816`, `mteb/trec-covid` `0724773f091de506941c2108e1647af956de07f7`.

| | TREC-COVID | NFCorpus | SciFact |
| --- | --- | --- | --- |
| Doküman | 171.332 | 3.633 | 5.183 |
| Sorgu | 50 (test) | 2.590 train / 324 dev / 323 test | 809 train / 300 test |
| Boyut (zip) | 73,9 MB | 2,4 MB | 2,8 MB |

**SciFact sızıntı temizliği (train için zorunlu).** Test pozitif dokümanlarının %64'ü (181/283) train'de de pozitif, 2 sorgu metni
train ve test'te ortak [ö]. "SciFact-temiz-train" şöyle üretilir ve çıkarılan id listesi `data/` altında sürümlenir (Y-54):
1. Train qrels'ten, dokümanı test pozitifi olan satırlar çıkarılır.
2. Test ile aynı metne sahip train sorguları çıkarılır.
3. Test bölümü değiştirilmez. Raporda iki sonuç birlikte verilir: tam test ve "181 örtüşen doküman çıkarılmış test".

Uygulama: `python3 data/scifact_temiz.py` (kayıt: `data/listeler/scifact_temiz_v1.json`; `--kontrol` ile makinedeki sonuç kayıtla karşılaştırılır).
Ölçülen sonuç [ö]: train 919 satır / 809 sorgu → **624 / 550**; temiz test 339 satır / 300 sorgu → **123 / 107**. 181 ortak dokümanın 295 train ve 216 test satırı çıktı, 2 sorgu metni ortak çıktı. **Temiz test küçüktür (107 sorgu):** küçük farkları yorumlama, tam test sonucunu ana sonuç, temiz testi sızıntı kontrolü olarak ver.

**NFCorpus:** yalnız zero-shot. Sorgu metni örtüşmesi var (train-test 4, train-dev 4, dev-test 1 [ö]); sonuç "doküman düzeyinde sızıntılı set" notuyla verilir. 40 yinelenen doküman var.

**TREC-COVID:** 42.140 dokümanın (%24,6) metni boş (yalnız başlık), 956 tam yinelenen doküman, 2 adet `-1` skoru var [ö]. Boş metinli dokümanda başlık kullanılır, `-1` satırları atılır. Yükleyicisi bu makinede 3 dakikada bitmedi [d].

**Lisans:** SciFact etiketleri CC BY 4.0, özetler ODC-By 1.0 [k]. TREC-COVID ve NFCorpus lisansı kaynak sayfasından **doğrulanamadı** [d] (Y-56). HF'deki `BeIR` kartları `cc-by-sa-4.0` yazıyor ama BEIR README bunun veri sahibinin lisansı olmadığını söylüyor [k].

## 4. TR-MTEB (SciFact-TR, NFCorpus-TR)

```python
from huggingface_hub import snapshot_download
for repo, rev in [
  ("trmteb/scifact-tr",  "e86dcd706c591848eed192fe6376a9ad4675854d"),
  ("trmteb/nfcorpus-tr", "c7e02f8114f4ca4ef9c2d47df193415928ef992e"),
]:
    snapshot_download(repo_id=repo, repo_type="dataset", revision=rev, local_dir=f"data/raw/{repo.split('/')[1]}")
```

- Corpus ve sorgu id'leri BEIR İngilizce ile aynı, qrels birebir aynı [ö]. Bu yüzden **bölüm 3'teki temizlik id listesi TR için de aynen uygulanır**.
- Çeviri: aya-expanse-8B; insan doğrulaması yalnız 115 örnekle yargıç kalibrasyonu için, tüm veri doğrulanmamış [k].
- Başlıklar çevrilmemiş (SciFact 5183/5183, NFCorpus 3633/3633 İngilizce aynı) [ö]. Başlık atılır ya da İngilizce olduğu belirtilir.
- Yükleme sonrası temizlik (Y-55): `<EOS_TOKEN>`, `<|im_end|>`, baştaki `Çeviri:` kaldırılır (NFCorpus'ta 132 sorgu ve 17 doküman, MsMarco'da ~33 passage etkilenmiş [ö]). Kırpılmış çeviriler (SciFact'te 25, NFCorpus'ta 13 doküman, uzunluğun yarısından kısa [ö]) işaretlenir.
- **`selmanbaysan/mteb_tr` fork'unun yükleme kodu kullanılmaz.** Test qrels'ten en çok 2048 satır alıp korpusu kısaltıyor ve sorgu başına tek ilgili doküman bırakıyor; sonuçlar BEIR ile karşılaştırılamaz. Kod okundu, çalıştırılmadı [k, d]. Kendi betiğimiz tam korpus ve tam qrels kullanır.
- **MsMarco-TR kullanılmaz:** test sorgularının 4.421'i train'de de var; korpus orijinalin yaklaşık %8'i; BEIR dev/test ile kesişimi sıfır, yani BEIR MS MARCO değil [ö]. Çeviri yöntemi belirsiz [d].
- Lisans: `trmteb/*` kartlarında lisans alanı yok, makale "yalnız ticari olmayan araştırma" diyor [k]. NFCorpus kaynağı akademik amaç dışında NutritionFacts şartlarına bağlı [k].
- Makalede yazılacaklar: makine çevirisi (aya-expanse-8B + LLM yargıç, 115 örnek), başlıkların çevrilmediği, temizlik adımları (Y-58).
- Türkçe sonuçlar tek başına "Türkçe yeteneği" iddiasına yetmez: doğal Türkçe veri değil, çeviri.

## 5. VectorDBBench Cohere 1M (yalnız DB testi)

```bash
B=https://assets.zilliz.com/benchmark/cohere_medium_1m
for f in test.parquet neighbors.parquet shuffle_train.parquet; do curl -fLO -C - $B/$f; done
# shuffle_train.parquet 3.134.418.448 B, ETag aa5dee664dc87b3e416c9863723a0ba4-374 (2023-06-09)
# sha256 (indir.py ile ölçüldü, 2026-11-07): test.parquet a2ec76c907da22e5533ce31c1105e4fd4d3c4597915a7771d5e41897a95ecbed
#                                          neighbors.parquet 30e7467766be3e2a6b167df16388a4a35e38fa6cc9e593fb659cce81ffde0940
# alternatif: pip install "vectordb-bench==2.0.0"   (VectorDBBench HEAD 9782fde, MIT)
```

- 1M x 768, float32, 1000 test sorgusu, sorgu başına 1000 komşu (ground truth). **Metin yok**, yalnız `id` ve `emb` [ö].
- **Metrik cosine olmalı.** Vektörler normalize değil (norm 12,3–17,5); cosine ve IP sıralaması 1000/1000 sorguda farklı [ö]. IP kullanılacaksa önce L2-normalize.
- `shuffle_train` satır sırası id sırası değil; id ile eşle. GT'nin 333.333 vektörlük örnekte exact cosine kNN ile tutarlı olduğu ölçüldü (recall@10 = 1,0). **Tam 1M üzerinde doğrulanmadı** [d]; Y-37'de exact kNN'i kendimiz hesaplarız.
- Maliyet (tahmin): disk ~7–10 GB, RAM zirvesi 6–9 GB. Batch ile yaz, GT'yi parça parça hesapla.
- Lisans: HF kartına erişilemedi, **doğrulanamadı** [d] (Y-56).
- Raporlanacaklar: QPS, p95 gecikme, indeks kurulum süresi, disk/RAM, recall@10/100. `benchmark.py` zaten `lancedb` dahil beş DB'yi içeriyor (`DBS`), Y-37 oraya bağlanır.

## 6. UDA (PaperTab + PaperText)

```bash
git clone https://github.com/qinchuanhui/UDA-Benchmark data/raw/UDA-Benchmark && git -C data/raw/UDA-Benchmark checkout fca5237ac316e776d8dbccffa55ca29c0efdc185
REV=d4367103fe8fe86b3bb76c66be8eafc4fb4117b2
curl -L -o paper_docs.zip "https://huggingface.co/datasets/qinchuanhui/UDA-QA/resolve/$REV/src_doc_files/paper_docs.zip"
echo "74890c04b6956c1a2f04adcd7946c0b9cc133a647ccc68854d6e9b5771b7a66a  paper_docs.zip" | shasum -a 256 -c -
unzip -q paper_docs.zip   # paper_docs/<arxiv_id>.pdf, 1129 dosya, ~0,98 GB
# sorular: UDA-Benchmark/dataset/qa/paper_tab_qa.csv ve paper_text_qa.csv
```

- Akademik olan yalnız PaperTab (307 PDF, 393 soru) ve PaperText (1087 PDF, 2804 soru); benzersiz 1129 PDF. UDA'da SQuAD yok [k, ö].
- 1129 PDF'in hepsi açıldı, her sorunun PDF'i mevcut, taranmış PDF yok, boş sayfa %0,12 [ö]. Ligatür ve çift sütun var, chunk öncesi **NFKC normalizasyonu**. Tablo ayrıştırma kalitesi ve PyPDF2 çıktısı **ölçülmedi** [d].
- Metrik: Qasper tarzı **token-F1**, birden çok referansta max. Cevapların ~%71'i abstractive, token-F1 kaba kalır. LLM-hakem eklemek ayrı karar (Y-57).
- **`my_eval.py` içindeki finans code-gen yolu LLM çıktısını `exec()` ile çalıştırır; o yol çağrılmaz.** Yalnız `basic_utils.token_f1_score` alınır.
- İlk turda sabit seed'li (42) ~300 soruluk alt küme. Alt küme id listesi `data/` altında sürümlenir (Y-57).
- Lisans: veri CC BY-SA 4.0, kaynak Qasper CC BY 4.0 [k]. **PDF telifi ayrı ve karışık:** rastgele 40 PDF'in 34'ü arXiv nonexclusive-distrib (yeniden dağıtım serbest değil), 4'ü CC BY 4.0, 2'si CC BY-NC-SA 4.0 [ö]; 1129'unun tamamı doğrulanamadı [d]. **PDF'ler repoya ve vault'a girmez.**
- Fine-tune için Qasper train kullanılacaksa UDA test PDF'leriyle arXiv id çakışması önce kontrol edilir [d].

## 7. Açık noktalar (tarifin bu sürümü bunları kapatmaz)

- SciFact train yalnız 809 sorgu / 919 qrels. Fine-tune ihtiyacı bundan büyükse ek train kaynağı için yeni karar gerekir (örn. MS MARCO İngilizce train, lisans: yalnız ticari olmayan araştırma).
- Çapraz-dil sızıntının skora etkisi ölçülmedi [d].
- MTEB kütüphanesiyle uyum ölçülmedi [d]; ortak betiğimiz kullanılır.
- Lisans doğrulamaları: TREC-COVID, NFCorpus, Cohere 1M, UDA PDF'leri (Y-56). Model ya da veri yayımlanmadan önce hukuki teyit gerekir; burada hukuki yorum yapılmadı.
