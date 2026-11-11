"""Metinleri vektöre çevirir ve her sorgu için en benzer dokümanları bulur (Y-55, ADIM 3b). İskelet.

Tarif kuralları (data/VERI-TARIFI.md §2): seed 42, doküman en çok 512 token, öneki ekle.
Bağımlılık: sentence-transformers, numpy (requirements.txt'e bak).
"""

MAKS_TOKEN = 512
SEED = 42


def kodla(model, metinler, onek="", batch=32):
    """-> numpy dizisi, şekil (len(metinler), boyut), her satır L2-normalize.

    TODO 1: her metnin başına `onek` ekle.
    TODO 2: model.max_seq_length = MAKS_TOKEN (kırpma burada olur).
    TODO 3: model.encode(...) ile vektörleri al; normalize_embeddings=True ile cosine = nokta çarpım olur.
    """
    raise NotImplementedError


def dokuman_metni(dokuman):
    """corpus kaydı ({"title","text"}) -> modele verilecek tek metin.

    TODO: başlık + boşluk + metin. Metin boşsa (TREC-COVID) yalnız başlık kullan.
    """
    raise NotImplementedError


def ara(sorgu_vek, dokuman_vek, dokuman_idleri, k=100):
    """-> list[list[str]]:  her sorgu için en benzer k doküman id'si, EN BENZERDEN başlayarak.

    TODO 1: skorlar = sorgu_vek @ dokuman_vek.T  (normalize olduğu için cosine).
    TODO 2: her satırda en yüksek k skorun indeksini al (numpy.argpartition hızlıdır, sonra sırala).
    TODO 3: indeksleri dokuman_idleri ile id'ye çevir.
    NOT:    TREC-COVID'de 171 bin doküman var; sorgu x doküman matrisi büyür, parça parça işle.
    """
    raise NotImplementedError
