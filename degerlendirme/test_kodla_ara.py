"""kodla_ara.py için kontrol (Y-55, adım 3b: kodla, dokuman_metni ve ara). Çalıştır:  python3 degerlendirme/test_kodla_ara.py

Gerçek model gerekmez: vektörler elle kurulur ya da SahteModel kullanılır, beklenen sonuç bilinir.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kodla_ara import MAKS_TOKEN, ara, dokuman_metni, kodla  # noqa: E402


class SahteModel:
    """SentenceTransformer gibi davranan basit nesne: encode() ve max_seq_length var.
    Gelen metinleri ve ayarları kaydeder, böylece kodla()'nın ne gönderdiğini kontrol edebiliriz."""

    max_seq_length = 8192  # gerçek modellerin geniş varsayılanını taklit eder (örn. BGE-M3)

    def encode(self, metinler, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        self.gelen = list(metinler)
        self.batch = batch_size
        self.normalize_istendi = normalize_embeddings
        # Her metin için [uzunluk, 1] vektörü; normalize istenirse uzunluğu 1 yapılır.
        vek = np.array([[float(len(m)), 1.0] for m in self.gelen])
        if normalize_embeddings:
            vek = vek / np.linalg.norm(vek, axis=1, keepdims=True)
        return vek


def vek_cosine(skor):
    """Sorguyla ([1, 0]) cosine'i tam `skor` olan, uzunluğu 1 bir vektör."""
    return [skor, math.sqrt(1 - skor * skor)]


# Konuşmadaki örnek: skorlar A=0.31, B=0.88, C=0.12, D=0.75, E=0.54 -> beklenen sıra B, D, E, A, C
SKORLAR = {"A": 0.31, "B": 0.88, "C": 0.12, "D": 0.75, "E": 0.54}
IDLER = list(SKORLAR)
DOK = np.array([vek_cosine(s) for s in SKORLAR.values()])
SORGU = np.array([[1.0, 0.0]])


def test_kodla_onek_eklenir():
    m = SahteModel()
    kodla(m, ["a", "bb"], onek="query: ")
    assert m.gelen == ["query: a", "query: bb"]


def test_kodla_onek_yoksa_metin_degismez():
    m = SahteModel()
    kodla(m, ["a", "bb"])
    assert m.gelen == ["a", "bb"]


def test_kodla_kirpma_siniri_512_ve_ozel():
    m = SahteModel()
    kodla(m, ["a"])
    assert m.max_seq_length == MAKS_TOKEN == 512
    kodla(m, ["a"], maks_token=2048)
    assert m.max_seq_length == 2048


def test_kodla_batch_iletilir():
    m = SahteModel()
    kodla(m, ["a"], batch=7)
    assert m.batch == 7


def test_kodla_sekil_tur_ve_normalize():
    m = SahteModel()
    vek = kodla(m, ["a", "bb", "ccc"])
    assert m.normalize_istendi is True
    assert vek.shape == (3, 2)
    assert vek.dtype == np.float32
    assert np.allclose(np.linalg.norm(vek, axis=1), 1.0, atol=1e-6)


def test_kodla_bos_liste_hata_verir():
    try:
        kodla(SahteModel(), [])
    except ValueError:
        return
    raise AssertionError("boş liste için ValueError beklenirdi")


def test_kodla_ve_ara_birlikte():
    # Sahte vektörler [uzunluk, 1] normalize: sorguyla aynı uzunluktaki metin en benzer çıkar.
    m = SahteModel()
    dok = kodla(m, ["aa", "aaaaaaaaaa", "aaaaa"])
    sorgu = kodla(m, ["bbbbb"])
    assert ara(sorgu, dok, ["kisa", "uzun", "orta"], k=3)[0][0] == "orta"


def test_baslik_ve_metin():
    assert dokuman_metni({"title": "Baslik", "text": "Govde."}) == "Baslik Govde."


def test_metin_bos_yalniz_baslik():
    assert dokuman_metni({"title": "Baslik", "text": ""}) == "Baslik"
    assert dokuman_metni({"title": "Baslik"}) == "Baslik"
    assert dokuman_metni({"title": "Baslik", "text": None}) == "Baslik"


def test_baslik_bos_yalniz_metin():
    assert dokuman_metni({"title": "", "text": "Govde."}) == "Govde."


def test_kenar_bosluklari_temizlenir():
    assert dokuman_metni({"title": " Baslik ", "text": " Govde. "}) == "Baslik Govde."


def test_ara_tam_siralama():
    assert ara(SORGU, DOK, IDLER, k=5) == [["B", "D", "E", "A", "C"]]


def test_ara_ilk_iki():
    assert ara(SORGU, DOK, IDLER, k=2) == [["B", "D"]]


def test_ara_k_dokuman_sayisindan_buyuk():
    assert ara(SORGU, DOK, IDLER, k=100) == [["B", "D", "E", "A", "C"]]


def test_ara_iki_sorgu():
    sorgular = np.array([[1.0, 0.0], [0.0, 1.0]])  # ikincisi birincinin tersini sever
    sonuc = ara(sorgular, DOK, IDLER, k=2)
    assert sonuc[0] == ["B", "D"]
    assert sonuc[1] == ["C", "A"]  # en düşük skorlular ikinci sorguda öne geçer


def test_ara_parca_sonucu_degistirmez():
    sorgular = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    assert ara(sorgular, DOK, IDLER, k=3, parca=1) == ara(sorgular, DOK, IDLER, k=3, parca=64)


def test_ara_esit_skorda_corpus_sirasi():
    dok = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert ara(SORGU, dok, ["d0", "d1", "d2"], k=2) == [["d0", "d1"]]


def test_ara_gecersiz_girdi():
    for k, dok_idleri in ((0, IDLER), (-1, IDLER), (2, IDLER[:3])):
        try:
            ara(SORGU, DOK, dok_idleri, k=k)
        except ValueError:
            continue
        raise AssertionError(f"ValueError beklenirdi: k={k}, id sayısı={len(dok_idleri)}")


if __name__ == "__main__":
    testler = [(ad, f) for ad, f in sorted(globals().items()) if ad.startswith("test_")]
    basarisiz = 0
    for ad, f in testler:
        try:
            f()
            print(f"  ok     {ad}")
        except Exception as e:  # noqa: BLE001
            basarisiz += 1
            print(f"  HATA   {ad}: {type(e).__name__}: {e}")
    print(f"{len(testler) - basarisiz}/{len(testler)} geçti")
    sys.exit(1 if basarisiz else 0)
