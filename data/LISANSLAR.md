# Veri seti lisansları

Durum: **kısmen doğrulandı.** Aşağıdaki bilgiler 2026-11-07 denetiminde kaynak sayfalarından okundu (alt agent, Claude);
"doğrulanamadı" yazanlar kaynak sayfasından okunamadı ve elle kontrol edilmeli (Y-56). Hukuki yorum değildir.
Model ya da veri yayımlanmadan önce hukuki teyit gerekir.

| Set | Lisans | Durum | Not |
| --- | --- | --- | --- |
| SciFact | Etiketler CC BY 4.0, korpus özetleri ODC-By 1.0 | kaynaktan okundu | `allenai/scifact` LICENSE.md |
| NFCorpus | "NFCorpus is free to use for academic purposes." Diğer kullanımlar için NutritionFacts.org Hizmet Şartları ve yazarı Dr. Michael Greger ile doğrudan iletişim | kaynaktan okundu | Ticari kullanım ya da yayımlama için yazardan izin gerekir. Kaynak: cl.uni-heidelberg.de/statnlpgroup/nfcorpus |
| TREC-COVID | NIST sayfasında lisans yok. Dayandığı CORD-19: Zenodo kaydında "Other (Open)", indirmek CORD-19 lisans sözleşmesini kabul etmek demek; makale başına lisans metadata dosyasında; ticari ve ticari olmayan alt kümeler ayrı | kısmen okundu | Sözleşme PDF'inin (COVID.DATA.LIC.AGMT.pdf) metni okunmadı. BEIR'deki TREC-COVID alt kümesinin makale başına lisansları incelenmedi. Kaynak: zenodo.org/records/3727291 |
| BEIR HF kartları (`BeIR/*`) | `cc-by-sa-4.0` etiketli | çelişkili | BEIR README bunun veri sahibinin lisansı olmadığını söylüyor; bağlayıcı olan kaynak lisansı |
| SciFact-TR, NFCorpus-TR (`trmteb/*`) | kartta lisans alanı yok; makale "yalnız ticari olmayan araştırma" diyor | kaynaktan okundu | Orijinal setin lisansı da geçerli |
| MsMarco-TR | yalnız ticari olmayan araştırma | kaynaktan okundu | Kapsam dışı (K-02) |
| VectorDBBench Cohere 1M | Veri için lisans **doğrulanamadı**. VectorDBBench kodu MIT | doğrulanamadı | VectorDBBench README'si veri lisansı vermiyor, kaynağı Wikipedia tabanlı `Cohere/wikipedia-22-12` olarak gösteriyor. HF kartları bu ortamdan açılmadı (401). Dolaylı iz: aynı ailenin bir kartı apache-2.0 diyor, ama bu setin kartı değil. **Elle bakılacak:** HF `Cohere/wikipedia-22-12-en-embeddings` kartı ve Wikipedia metni için CC BY-SA yükümlülükleri |
| UDA veri seti | CC BY-SA 4.0 | kaynaktan okundu | Kaynak Qasper CC BY 4.0 |
| UDA içindeki PDF'ler | **karışık**, 1129 PDF'in tamamı arXiv metadata'sından okundu | kaynaktan okundu | Aşağıdaki tablo. Makale başına liste: `data/listeler/uda_pdf_lisanslari_v1.json`. **PDF'ler repoya ve vault'a girmez.** |

## UDA PDF lisansları (Y-56, 2026-11-11)

Her PDF'in lisansı arXiv'in OAI-PMH arayüzünden, makale numarasıyla sorgulandı (1129 sorgu, hata yok). Aşağıdaki sayılar arXiv'in kayıtlı lisansıdır; yazarın sonradan yaptığı değişiklik ya da yayıncı şartı burada görünmez.

| Lisans | PDF sayısı | Yeniden dağıtım |
| --- | --- | --- |
| arXiv nonexclusive-distrib 1.0 | 984 | Serbest değil, arXiv'e dağıtım izni verir |
| CC BY 4.0 | 81 | Serbest, atıfla |
| CC BY-NC-SA 4.0 | 38 | Yalnız ticari olmayan, aynı lisansla |
| CC BY-SA 4.0 | 16 | Serbest, atıfla ve aynı lisansla |
| CC0 1.0 | 8 | Serbest |
| CC BY-NC-SA 3.0 | 1 | Yalnız ticari olmayan, aynı lisansla |
| CC BY-NC-ND 4.0 | 1 | Ticari olmayan, türev yok |

Sonuç: 1129 PDF'in %87,2'si (984) yeniden dağıtıma kapalı, 105'i (%9,3) serbest, 40'ı yalnız ticari olmayan kullanıma açık. Bu yüzden PDF'ler hiçbir yerde yayımlanmaz; yalnız `q_uid` ve makale numarası listeleri paylaşılır. Önceki 40 PDF'lik örneklem (34 / 4 / 2) bu dağılımla uyumluydu. Hukuki yorum değildir.
