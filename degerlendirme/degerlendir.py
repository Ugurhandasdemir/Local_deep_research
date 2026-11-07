#!/usr/bin/env python3
"""Ortak değerlendirme betiği: tek komutla bir model + bir set -> bir sonuç satırı (Y-55, ADIM 4). İskelet.

Hedef kullanım:
    python3 degerlendirme/degerlendir.py --model intfloat/multilingual-e5-base --set scifact --bolum test
    python3 degerlendirme/degerlendir.py --model ... --set scifact --qrels data/raw/scifact_temiz/test_temiz.tsv

Akış (her adım ilgili dosyada):
    beir_yukle.yukle -> onekler.onek_al -> kodla_ara.kodla (dokümanlar, sorgular) -> kodla_ara.ara
    -> metrikler.ndcg_at_k / recall_at_k (her sorgu için, sonra ortalama) -> satir_yaz
"""
import argparse
import sys


def sonuc_satiri(set_adi, bolum, model, ndcg10, recall100, sorgu_sayisi, seed, tarif_surumu):
    """-> dict: tek sonuç satırı.

    TODO: tarif §2 madde 6'daki alanları ekle: tarif sürümü, set + bölüm, model, seed, donanım
          (platform.platform(), torch cihazı), kütüphane sürümleri (sentence_transformers, torch),
          sorgu sayısı, nDCG@10, Recall@100, zaman damgası.
    """
    raise NotImplementedError


def degerlendir(model_adi, set_klasoru, bolum="test", qrels_yolu=None):
    """-> dict: sonuç satırı.

    TODO: dosya başlığındaki akışı sırayla uygula. Sorgu başına nDCG@10 ve Recall@100'ü hesapla,
          ortalamasını al. Doğrusu boş sorguyu (ValueError) hiç hesaba katma ve KAÇ sorgu
          atlandığını sonuç satırına yaz.
    """
    raise NotImplementedError


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ortak retrieval değerlendirmesi (Y-55).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", dest="set_adi", required=True, help="scifact | nfcorpus | trec-covid")
    ap.add_argument("--bolum", default="test")
    ap.add_argument("--qrels", default=None, help="temiz qrels yolu (Y-54)")
    ap.add_argument("--cikti", default=None, help="sonuç satırını JSON olarak ekle")
    ap.parse_args(argv)
    # TODO: degerlendir'i çağır, satırı yazdır (ve --cikti verildiyse dosyaya ekle).
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
