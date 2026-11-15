"""metrikler.py için kontrol (Y-55, adım 1). Çalıştır:  python3 degerlendirme/test_metrikler.py

Beklenen sayılar elle hesaplanmış örnekten gelir (3 doğru doküman: D1, D2, D3).
Bu dosyayı değiştirme; yazdığın fonksiyon bunlara uymalı.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrikler import ndcg_at_k, recall_at_k  # noqa: E402

DOGRU = {"D1", "D2", "D3"}
IYI = ["D1", "x", "D2", "x", "x", "D3", "x", "x", "x", "x"]    # doğrular 1., 3., 6. sırada
KOTU = ["x", "x", "x", "x", "x", "x", "x", "x", "x", "D1"]     # tek doğru 10. sırada


def yakin(a, b):
    return math.isclose(a, b, abs_tol=5e-4)


def test_ndcg_iyi():
    assert yakin(ndcg_at_k(IYI, DOGRU, 10), 0.871)


def test_ndcg_kotu():
    assert yakin(ndcg_at_k(KOTU, DOGRU, 10), 0.136)


def test_mukemmel_siralama_bir_eder():
    assert yakin(ndcg_at_k(["D1", "D2", "D3", "x"], DOGRU, 10), 1.0)


def test_hic_dogru_yoksa_sifir():
    assert ndcg_at_k(["x", "y", "z"], DOGRU, 10) == 0.0


def test_k_dan_sonrasi_sayilmaz():
    # D1 yalnız 10. sırada; k=5 ile bakınca görünmez.
    assert ndcg_at_k(KOTU, DOGRU, 5) == 0.0


def test_dogru_sayisi_k_dan_buyukse_ideal_k_ile_sinirli():
    # 5 doğru doküman var ama k=3: ilk 3'ü hepsi doğruysa bu MÜKEMMELDİR (1.0).
    # İdeal puan en fazla k terim toplar; 5 terim toplarsan sonuç 1.0'ın altında kalır.
    bes = {"A", "B", "C", "D", "E"}
    assert yakin(ndcg_at_k(["A", "B", "C", "x", "y"], bes, 3), 1.0)


def test_siralama_k_dan_kisaysa_hata_vermez():
    assert 0.0 <= ndcg_at_k(["D1"], DOGRU, 10) <= 1.0


def test_recall():
    assert yakin(recall_at_k(IYI, DOGRU, 10), 1.0)
    assert yakin(recall_at_k(KOTU, DOGRU, 10), 1 / 3)
    assert yakin(recall_at_k(IYI, DOGRU, 3), 2 / 3)  # ilk 3'te D1 ve D2 var


def test_dogru_listesi_bossa_hata_verir():
    # Sessizce 0 dönmek ortalamayı bozar. Böyle sorgu hesaba hiç girmemeli (tarif: "paydada kim var").
    for fn in (ndcg_at_k, recall_at_k):
        try:
            fn(IYI, set(), 10)
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__}: boş doğru kümesinde ValueError beklenirdi")


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
