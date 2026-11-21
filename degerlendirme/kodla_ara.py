"""Metinleri vektöre çevirir ve her sorgu için en benzer dokümanları bulur (Y-55, ADIM 3b).

Durum: kodla(), dokuman_metni() ve ara() hazır ve sahte modelle test edildi (test_kodla_ara.py).
kodla() gerçek bir SentenceTransformer ile henüz denenmedi (bu Mac'te kurulu değil, Colab'da denenecek).

Tarif kuralları (data/VERI-TARIFI.md §2): seed 42, doküman en çok 512 token, öneki ekle.
Bağımlılık: sentence-transformers, numpy (requirements.txt'e bak).
"""
import numpy as np

MAKS_TOKEN = 512
SEED = 42


def kodla(model, metinler, onek="", batch=32, maks_token=MAKS_TOKEN):
    """-> numpy dizisi, şekil (len(metinler), boyut), her satır L2-normalize.

    model: sentence-transformers modeli (SentenceTransformer). Test için encode() ve max_seq_length'i
           olan her nesne olur, gerçek model şart değil.
    onek: her metnin başına eklenir (onekler.onek_al ile alınır). Bu fonksiyon yalnız metin önekli
          modeller içindir; Jina v3 (task=) ve EmbeddingGemma 2 (istem adı) ayrı destek ister.
    maks_token: kırpma sınırı. Tarif 512 der (varsayılan); Y-36'daki "daha uzun sınırla ikinci koşu"
                kararı için başka bir sayı verilebilir.
    """
    if len(metinler) == 0:
        raise ValueError("metin listesi boş")
    # Liste üreteci: her metnin önüne öneki ekler (onek "" ise metin değişmez).
    metinler = [onek + metin for metin in metinler]
    # Kırpma burada olur: model bu sayıdan uzun girdiyi keser. Model nesnesi kendi sınırını taşır
    # (örn. BGE-M3 varsayılan 8192), o yüzden her çağrıda açıkça ayarlıyoruz.
    model.max_seq_length = maks_token
    vektorler = model.encode(
        metinler,
        batch_size=batch,                # metinler 32'şerli gruplar hâlinde modele girer
        normalize_embeddings=True,       # her vektörü uzunluk 1 yapar, cosine = nokta çarpım olur
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    # float32: bellek yarıya iner, ara() için yeterli doğruluk.
    return np.asarray(vektorler, dtype=np.float32)


def dokuman_metni(dokuman):
    """corpus kaydı ({"title","text"}) -> modele verilecek tek metin.

    Başlık + boşluk + metin. Metin boşsa (TREC-COVID) yalnız başlık, başlık boşsa yalnız metin.
    """
    # dict.get("title"): anahtar yoksa hata vermez, None döner.
    # `x or ""`: x None ya da boşsa "" verir (None üzerinde .strip() çağıramayız).
    baslik = (dokuman.get("title") or "").strip()
    metin = (dokuman.get("text") or "").strip()
    # " ".join(...): parçaları aralarına boşluk koyarak birleştirir.
    # `for p in (...) if p`: yalnız dolu olanları alır, böylece boş parça fazladan boşluk bırakmaz.
    return " ".join(p for p in (baslik, metin) if p)


def ara(sorgu_vek, dokuman_vek, dokuman_idleri, k=100, parca=64):
    """-> list[list[str]]:  her sorgu için en benzer k doküman id'si, EN BENZERDEN başlayarak.

    sorgu_vek: (sorgu_sayisi, boyut), dokuman_vek: (doküman_sayisi, boyut); ikisi de L2-normalize olmalı,
    böylece nokta çarpım = cosine olur.
    parca: aynı anda kaç sorgunun skorunun hesaplanacağı. TREC-COVID'de 171 bin doküman var, tüm sorguları
    birden çarparsak skor matrisi belleğe sığmaz; sorguları parça parça işleriz.
    Seçilen k aday içinde eşit skorda corpus sırası öne geçer. k sınırında eşit skorlu birden çok doküman varsa
    hangisinin girdiğini argpartition belirler (nadir, tüm skorların birebir eşit olması gerekir).
    """
    if k <= 0:
        raise ValueError(f"k pozitif olmalı, gelen: {k}")
    if len(dokuman_vek) != len(dokuman_idleri):
        raise ValueError(
            f"doküman vektörü ({len(dokuman_vek)}) ile id listesi ({len(dokuman_idleri)}) aynı uzunlukta olmalı"
        )
    k = min(k, len(dokuman_idleri))  # k doküman sayısından büyükse hepsini döndür

    sonuc = []
    for bas in range(0, len(sorgu_vek), parca):
        # sorgu_vek[bas:bas + parca]: bu parçadaki sorgular. "@" matris çarpımı.
        # skorlar şekli: (parçadaki sorgu sayısı, doküman sayısı).
        skorlar = sorgu_vek[bas:bas + parca] @ dokuman_vek.T
        for satir in skorlar:  # satir = tek bir sorgunun tüm dokümanlarla skorları
            # argpartition en küçük k değeri öne toplar (sıralamaz, hızlıdır). En BÜYÜK k'yı istediğimiz için
            # skorları eksiye çeviririz (-satir): en büyük skor en küçük olur.
            aday = np.argpartition(-satir, k - 1)[:k]
            aday = np.sort(aday)  # eşit skorda corpus sırası kazansın diye dokümanları sıraya diz
            # Yalnız k adayı sırala. kind="stable": eşitlerde mevcut sırayı bozma.
            sirali = aday[np.argsort(-satir[aday], kind="stable")]
            # Liste üreteci: indeksleri doküman id'sine çevirir.
            sonuc.append([dokuman_idleri[i] for i in sirali])
    return sonuc
