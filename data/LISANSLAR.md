# Veri seti lisansları

Durum: **kısmen doğrulandı.** Aşağıdaki bilgiler 2026-11-07 denetiminde kaynak sayfalarından okundu (alt agent, Claude);
"doğrulanamadı" yazanlar kaynak sayfasından okunamadı ve elle kontrol edilmeli (Y-56). Hukuki yorum değildir.
Model ya da veri yayımlanmadan önce hukuki teyit gerekir.

| Set | Lisans | Durum | Not |
| --- | --- | --- | --- |
| SciFact | Etiketler CC BY 4.0, korpus özetleri ODC-By 1.0 | kaynaktan okundu | `allenai/scifact` LICENSE.md |
| NFCorpus | "Akademik amaçlarla serbest, başka kullanım için NutritionFacts Hizmet Şartları" | kaynaktan okundu | Ticari kullanım koşullu |
| TREC-COVID | — | **doğrulanamadı** | NIST sayfasından lisans bilgisi çıkmadı |
| BEIR HF kartları (`BeIR/*`) | `cc-by-sa-4.0` etiketli | çelişkili | BEIR README bunun veri sahibinin lisansı olmadığını söylüyor; bağlayıcı olan kaynak lisansı |
| SciFact-TR, NFCorpus-TR (`trmteb/*`) | kartta lisans alanı yok; makale "yalnız ticari olmayan araştırma" diyor | kaynaktan okundu | Orijinal setin lisansı da geçerli |
| MsMarco-TR | yalnız ticari olmayan araştırma | kaynaktan okundu | Kapsam dışı (K-02) |
| VectorDBBench Cohere 1M | — | **doğrulanamadı** | HF kartına erişilemedi (401). VectorDBBench kodu MIT. Wikipedia kaynaklı metin genelde CC BY-SA ama bu setin kartından doğrulanmadı |
| UDA veri seti | CC BY-SA 4.0 | kaynaktan okundu | Kaynak Qasper CC BY 4.0 |
| UDA içindeki PDF'ler | **karışık** | kısmen ölçüldü | Rastgele 40 PDF: 34 arXiv nonexclusive-distrib (yeniden dağıtım serbest değil), 4 CC BY 4.0, 2 CC BY-NC-SA 4.0. 1129 PDF'in tamamı doğrulanmadı. **PDF'ler repoya ve vault'a girmez.** |
