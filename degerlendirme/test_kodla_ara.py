"""kodla_ara.py için kontrol (Y-55, adım 3b: dokuman_metni ve ara). Çalıştır:  python3 degerlendirme/test_kodla_ara.py

Gerçek model gerekmez: vektörler elle kurulur, böylece beklenen sıralama bilinir.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kodla_ara import ara, dokuman_metni  # noqa: E402


def vek_cosine(skor):
    """Sorguyla ([1, 0]) cosine'i tam `skor` olan, uzunluğu 1 bir vektör."""
    return [skor, math.sqrt(1 - skor * skor)]


# Konuşmadaki örnek: skorlar A=0.31, B=0.88, C=0.12, D=0.75, E=0.54 -> beklenen sıra B, D, E, A, C
SKORLAR = {"A": 0.31, "B": 0.88, "C": 0.12, "D": 0.75, "E": 0.54}
IDLER = list(SKORLAR)
DOK = np.array([vek_cosine(s) for s in SKORLAR.values()])
SORGU = np.array([[1.0, 0.0]])


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
