"""onekler.py için kontrol (Y-55, adım 3a). Çalıştır:  python3 degerlendirme/test_onekler.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onekler import ONEKLER, onek_al  # noqa: E402


def test_e5_sorgu_ve_dokuman():
    assert onek_al("intfloat/multilingual-e5-base") == ("query: ", "passage: ")


def test_bge_m3_onek_yok():
    assert onek_al("BAAI/bge-m3") == ("", "")


def test_qwen3_sadece_sorguya_talimat():
    sorgu, dokuman = onek_al("Qwen/Qwen3-Embedding-0.6B")
    assert sorgu.startswith("Instruct: ")
    assert sorgu.endswith("\nQuery:")
    assert dokuman == ""


def test_qwen3_uc_boyut_ayni_sablon():
    assert onek_al("Qwen/Qwen3-Embedding-4B") == onek_al("Qwen/Qwen3-Embedding-0.6B")
    assert onek_al("Qwen/Qwen3-Embedding-8B") == onek_al("Qwen/Qwen3-Embedding-0.6B")


def test_bilinmeyen_model_hata_verir():
    try:
        onek_al("jinaai/jina-embeddings-v3")
    except KeyError as e:
        assert "önek tanımlı değil" in str(e)
    else:
        raise AssertionError("tanımsız model için KeyError beklenirdi")


def test_her_kayitta_kaynak_var():
    for ad, kayit in ONEKLER.items():
        assert kayit["kaynak"].startswith("https://"), ad


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
