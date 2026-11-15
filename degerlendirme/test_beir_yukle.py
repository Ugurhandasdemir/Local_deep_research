"""beir_yukle.py için kontrol (Y-55, adım 2). Çalıştır:  python3 degerlendirme/test_beir_yukle.py

Küçük, elle yazılmış sahte bir BEIR klasörü kurar; gerçek veri indirmez.
Gerçek SciFact ile deneme için dosyanın sonundaki `--gercek` seçeneğine bak.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beir_yukle import yukle  # noqa: E402


def sahte_set(klasor: Path):
    (klasor / "qrels").mkdir(parents=True)
    corpus = [
        {"_id": "D1", "title": "Başlık bir", "text": "ilk metin"},
        {"_id": "D2", "title": "Başlık iki", "text": ""},          # boş metin
        {"_id": "D3", "text": "başlıksız doküman"},                # title yok
    ]
    queries = [
        {"_id": "q1", "text": "Türkçe İ ı karakterli sorgu"},
        {"_id": "q2", "text": "ikinci sorgu"},
        {"_id": "q3", "text": "yalnız train'de olan sorgu"},
    ]
    (klasor / "corpus.jsonl").write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in corpus) + "\n", encoding="utf-8")
    (klasor / "queries.jsonl").write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in queries) + "\n", encoding="utf-8")
    (klasor / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\tD1\t1\nq1\tD2\t1\nq2\tD3\t0\nq2\tD1\t1\n", encoding="utf-8")
    (klasor / "qrels" / "train.tsv").write_text("query-id\tcorpus-id\tscore\nq3\tD3\t1\n", encoding="utf-8")


def test_yukle_sekilleri():
    with tempfile.TemporaryDirectory() as t:
        sahte_set(Path(t))
        corpus, sorgular, qrels = yukle(t, "test")
        assert set(corpus) == {"D1", "D2", "D3"}
        assert corpus["D3"]["title"] == ""            # title yoksa ""
        assert corpus["D2"]["text"] == ""             # boş metin korunur, doküman atılmaz
        assert qrels == {"q1": {"D1", "D2"}, "q2": {"D1"}}   # D3'ün skoru 0, doğru sayılmaz
        assert set(sorgular) == {"q1", "q2"}          # q3 test'te yok, alınmaz
        assert sorgular["q1"] == "Türkçe İ ı karakterli sorgu"   # Türkçe karakter bozulmadı


def test_baska_bolum():
    with tempfile.TemporaryDirectory() as t:
        sahte_set(Path(t))
        _, sorgular, qrels = yukle(t, "train")
        assert qrels == {"q3": {"D3"}} and set(sorgular) == {"q3"}


def test_ozel_qrels_yolu():
    with tempfile.TemporaryDirectory() as t:
        sahte_set(Path(t))
        ozel = Path(t) / "temiz.tsv"
        ozel.write_text("query-id\tcorpus-id\tscore\nq2\tD1\t1\n", encoding="utf-8")
        _, sorgular, qrels = yukle(t, qrels_yolu=ozel)
        assert qrels == {"q2": {"D1"}} and set(sorgular) == {"q2"}


def test_dosya_yoksa_anlasilir_hata():
    with tempfile.TemporaryDirectory() as t:
        try:
            yukle(t, "test")
        except FileNotFoundError as e:
            assert "indir.py" in str(e)
            return
        raise AssertionError("FileNotFoundError beklenirdi")


def test_bozuk_baslik_hata_verir():
    with tempfile.TemporaryDirectory() as t:
        sahte_set(Path(t))
        (Path(t) / "qrels" / "test.tsv").write_text("a\tb\tc\nq1\tD1\t1\n", encoding="utf-8")
        try:
            yukle(t, "test")
        except ValueError:
            return
        raise AssertionError("ValueError beklenirdi")


def test_qrels_te_olup_sorgusu_olmayan_hata_verir():
    with tempfile.TemporaryDirectory() as t:
        sahte_set(Path(t))
        (Path(t) / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq99\tD1\t1\n", encoding="utf-8")
        try:
            yukle(t, "test")
        except ValueError as e:
            assert "q99" in str(e)
            return
        raise AssertionError("ValueError beklenirdi")


def gercek_scifact(klasor):
    """Elle deneme: python3 degerlendirme/test_beir_yukle.py --gercek data/raw/scifact"""
    corpus, sorgular, qrels = yukle(klasor, "test")
    print(f"doküman={len(corpus)} sorgu={len(sorgular)} qrels sorgusu={len(qrels)} "
          f"doğru çift={sum(len(v) for v in qrels.values())}")
    print("beklenen (SciFact test): doküman=5183 sorgu=300 qrels sorgusu=300 doğru çift=339")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--gercek":
        gercek_scifact(sys.argv[2])
        sys.exit(0)
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
