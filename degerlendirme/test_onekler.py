"""onekler.py için kontrol (Y-55, adım 3a). Çalıştır:  python3 -m pytest -q degerlendirme

Beklenen değerler model kartlarından okunmuştur (kaynak bağlantıları onekler.py'de).
Bu dosyayı değiştirme; yazdığın sözlük ve fonksiyonlar bunlara uymalı.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onekler import ONEKLER, onek_al, yukle_ayarlari  # noqa: E402

GECERLI_YONTEMLER = {"metin", "model", "task"}

QWEN_SORGU = (
    "Instruct: Given a web search query, "
    "retrieve relevant passages that answer the query\nQuery:"
)


# --- onek_al: her yöntem için bir örnek ---

def test_metin_yontemi_e5():
    assert onek_al("intfloat/multilingual-e5-base") == ("metin", "query: ", "passage: ")


def test_none_bos_metne_cevrilir():
    # BGE-M3 kartı: önek gerekmiyor. None, bilerek "" olarak döner.
    assert onek_al("BAAI/bge-m3") == ("metin", "", "")


def test_qwen_talimati_yalniz_sorguda():
    yontem, sorgu, dokuman = onek_al("Qwen/Qwen3-Embedding-0.6B")
    assert yontem == "metin"
    assert sorgu == QWEN_SORGU
    assert "\nQuery:" in sorgu and not sorgu.endswith(" ")
    assert dokuman == ""


@pytest.mark.parametrize("model", ["google/embeddinggemma-300m", "magibu/embeddingmagibu2"])
def test_model_yontemi_metin_vermez(model):
    # Önek metnini kütüphane ekler (encode_query / encode_document), bizde değer yok.
    assert onek_al(model) == ("model", None, None)


def test_task_yontemi_jina():
    assert onek_al("jinaai/jina-embeddings-v3") == (
        "task", "retrieval.query", "retrieval.passage",
    )


def test_bilinmeyen_model_anlasilir_hata_verir():
    with pytest.raises(ValueError) as hata:
        onek_al("olmayan/model")
    assert "olmayan/model" in str(hata.value)


# --- yukle_ayarlari ---

def test_yukle_ayarlari_yoksa_bos():
    assert yukle_ayarlari("BAAI/bge-m3") == {}


def test_yukle_ayarlari_jina_ve_magibu():
    assert yukle_ayarlari("jinaai/jina-embeddings-v3") == {"trust_remote_code": True}
    assert yukle_ayarlari("magibu/embeddingmagibu2") == {
        "config_kwargs": {"vision_config": None, "audio_config": None}
    }


# --- sözlüğün tamamı için tutarlılık (yeni model eklenince de çalışır) ---

def test_altı_model_var():
    assert set(ONEKLER) == {
        "intfloat/multilingual-e5-base",
        "BAAI/bge-m3",
        "Qwen/Qwen3-Embedding-0.6B",
        "google/embeddinggemma-300m",
        "magibu/embeddingmagibu2",
        "jinaai/jina-embeddings-v3",
    }


@pytest.mark.parametrize("model", sorted(ONEKLER))
def test_her_kayit_tam(model):
    kayit = ONEKLER[model]
    assert kayit["yontem"] in GECERLI_YONTEMLER
    assert kayit["kaynak"] == f"https://huggingface.co/{model}"
    assert "sorgu" in kayit and "dokuman" in kayit


@pytest.mark.parametrize("model", sorted(ONEKLER))
def test_onek_al_her_model_icin_calisir(model):
    yontem, sorgu, dokuman = onek_al(model)
    assert yontem == ONEKLER[model]["yontem"]
    if yontem == "metin":
        # Metin yönteminde asla None dönmez; yanlışlıkla "None" metni eklenmesin.
        assert isinstance(sorgu, str) and isinstance(dokuman, str)
