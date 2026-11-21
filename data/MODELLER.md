# Karşılaştırılacak embedding modelleri (Y-41, Y-03)

Durum: **kartlardan okundu, 2026-11-21.** Hukuki yorum değildir. Model yayımlanmadan ya da ticari kullanılmadan önce lisans metni ayrıca okunmalı.
Önek ya da talimat verilmezse model hata vermez, ama sıralama kötüleşir. Bu yüzden her model için aşağıdaki biçim kullanılır.

## Özet tablo

| Model | HF kimliği | Parametre | Vektör boyutu | Lisans | Sorgu ve belge biçimi |
| --- | --- | --- | --- | --- | --- |
| Qwen3-Embedding | `Qwen/Qwen3-Embedding-0.6B`, `-4B`, `-8B` | 0,6B, 4B, 8B | 1024, 2560, 4096 (hepsi esnek boyutu destekler) | Apache-2.0 | Yalnız sorguya talimat eklenir, belge olduğu gibi girer |
| EmbeddingGemma 2 | `google/embeddinggemma-2` | 740M (metin kısmı 270M) | 768 (128, 256, 512'ye kısaltılabilir) | Apache-2.0 | Sorguya `SearchQuery`, belgeye `Document` istemi, başlıklı belge `title: {başlık} \| text: {metin}`, başlıksızsa `title: none` |
| BGE-M3 | `BAAI/bge-m3` | bekliyor | 1024, bağlam 8192 token | MIT | Talimat gerekmiyor, sorgu ve belge düz metin |
| Jina Embeddings v3 | `jinaai/jina-embeddings-v3` | bekliyor | 1024'e kadar (32 ile 1024 arası kısaltılabilir) | **CC BY-NC 4.0** (ticari olmayan) | `task="retrieval.query"` ile sorgu, `task="retrieval.passage"` ile belge |
| Multilingual E5 Base | `intfloat/multilingual-e5-base` | bekliyor | 768 (12 katman) | MIT | Sorguya `query: `, belgeye `passage: ` ön eki, en fazla 512 token |

## Kaynaklar (kart sayfaları ve ham README'ler)

- Qwen3: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B (boyut tablosu, lisans meta verisi `apache-2.0`; 4B ve 8B kartlarında da aynı)
- EmbeddingGemma 2: https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2 (lisans Apache 2.0, 740M, 768 boyut, 8192 bağlam, `SearchQuery` ve `Document` istemleri)
- BGE-M3: https://huggingface.co/BAAI/bge-m3 (lisans meta verisi `mit`; "BGE-M3 modeli artık sorgulara talimat eklenmesini gerektirmiyor")
- Jina v3: https://huggingface.co/jinaai/jina-embeddings-v3 (lisans meta verisi `cc-by-nc-4.0`; iki görev adı)
- E5: https://huggingface.co/intfloat/multilingual-e5-base (lisans meta verisi `mit`; "her giriş `query: ` ya da `passage: ` ile başlamalı")

## Qwen3 sorgu şablonu

Karttaki örnek işlev, talimatı sorgunun önüne ekler:

```
Instruct: {görev açıklaması}
Query:{sorgu}
```

Kartta görev açıklaması örnek olarak bir cümlelik ("Given a web search query, retrieve relevant passages that answer the query" türünden). Belge metni talimatsız girer. Kartın kendisi "her sorgu bir cümlelik talimatla gelmeli" der.

## Dikkat edilecekler

- **Jina v3 ticari olmayan lisanslı.** Makalede sonuçlar ayrı işaretlenmeli, ticari kullanıma girilecekse model listeden çıkarılmalı. AWS ve Azure üzerinden ticari kullanım ayrı bir kanaldır, bizim kullanımımız değil.
- **EmbeddingGemma'nın iki sürümü var.** Eski `google/embeddinggemma-300m` kartı kapılı (`gemma` lisansı, ham dosya 401 verdi), bu çalışmada okunamadı. Yeni sürüm kartı Apache-2.0. Y-41 "EmbeddingGemma 2" dediği için bu tabloda yeni sürüm esas alındı.
- **EmbeddingGemma 2 "multimodal".** Metin dışında görüntü, video ve ses de alıyor. Parametre sayısı 740M, bunun yalnız 270M'i metin kısmı. Bellek hesabında toplam sayı geçerli olabilir, bu henüz ölçülmedi.
- **VRAM (çıkarım, ölçülmedi):** hedef makinede 4 GB VRAM var. Qwen3 4B ve 8B modellerinin 4 GB'a sığması beklenmiyor. Quantization ya da CPU gerekebilir, bunun ayrıca kararı gerekiyor.

## Açık (bu PR'da doğrulanmadı)

- BGE-M3, Jina v3 ve E5-base için toplam parametre sayısı kartta bulunamadı.
- EmbeddingGemma 2'nin `SearchQuery` ve `Document` istemlerinin tam metni (sentence-transformers istem adıyla kullanılır, metin elle yazılmaz).
- Eski `embeddinggemma-300m` kartının lisans ve önek bilgisi (kapılı).
