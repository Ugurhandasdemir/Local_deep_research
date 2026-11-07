#!/usr/bin/env python3
"""SciFact'i sızıntıdan arındırır (Y-54, tarif: data/VERI-TARIFI.md §3).

Sorun: SciFact'te test sorgularının doğru cevabı olan dokümanların %64'ü (181/283) train
sorgularının da doğru cevabı. Model train'de bu dokümanı "ilgili" diye ezberlerse test'te
haksız avantaj kazanır. Bu betik üç şey üretir:

  1. train_temiz.tsv : train qrels'ten çıkarılanlar sonrası (fine-tune YALNIZ bununla yapılır)
       - dokümanı herhangi bir test pozitifi olan satırlar çıkar
       - test sorgularıyla aynı metne sahip train sorgularının tüm satırları çıkar
  2. test_temiz.tsv  : test qrels'ten, dokümanı train pozitifi olan satırlar çıkar.
       Raporda tam test (qrels/test.tsv, DEĞİŞMEZ) ile birlikte verilir.
  3. scifact_temiz_v1.json : neyin çıkarıldığının kaydı (repoya girer, üçümüz aynı sonucu
       aldığımızı `--kontrol` ile doğrularız).

Kullanım:
    python3 data/indir.py beir                       # önce SciFact'i indir
    python3 data/scifact_temiz.py                    # data/raw/scifact_temiz/ altına yazar
    python3 data/scifact_temiz.py --kontrol          # çıktı, repodaki listeyle aynı mı?

Yalnızca standart kütüphane. Çıktı deterministiktir (sıralı), rastgelelik yok.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent
VARSAYILAN_GIRDI = DATA / "raw" / "scifact"
VARSAYILAN_CIKTI = DATA / "raw" / "scifact_temiz"
LISTE_YOLU = DATA / "listeler" / "scifact_temiz_v1.json"
SURUM = "scifact_temiz_v1"  # kural değişirse v2 aç, v1'i silme (tarif kuralı)


def qrels_oku(yol: Path):
    """[(sorgu_id, doküman_id, skor)] döner. Başlık satırı atlanır."""
    with open(yol, encoding="utf-8", newline="") as f:
        satirlar = list(csv.reader(f, delimiter="\t"))
    if not satirlar or satirlar[0] != ["query-id", "corpus-id", "score"]:
        raise ValueError(f"{yol}: beklenen başlık yok (query-id, corpus-id, score)")
    return [(q, d, int(s)) for q, d, s in satirlar[1:]]


def qrels_yaz(yol: Path, satirlar) -> None:
    yol.parent.mkdir(parents=True, exist_ok=True)
    with open(yol, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["query-id", "corpus-id", "score"])
        w.writerows(satirlar)


def sorgu_metinleri(yol: Path) -> dict:
    metin = {}
    with open(yol, encoding="utf-8") as f:
        for satir in f:
            o = json.loads(satir)
            metin[o["_id"]] = o["text"]
    return metin


def normalle(metin: str) -> str:
    # Büyük/küçük harf ve boşluk farkı aynı sorguyu başka göstermesin.
    return " ".join(metin.lower().split())


def temizle(girdi: Path) -> dict:
    train = qrels_oku(girdi / "qrels" / "train.tsv")
    test = qrels_oku(girdi / "qrels" / "test.tsv")
    metin = sorgu_metinleri(girdi / "queries.jsonl")

    # Pozitif = skor > 0. SciFact'te tüm skorlar 1, ama kuralı açık yazmak daha güvenli.
    train_poz = {d for _, d, s in train if s > 0}
    test_poz = {d for _, d, s in test if s > 0}
    ortak_dokuman = train_poz & test_poz

    # Kural 2: test'tekiyle aynı metne sahip train sorguları.
    test_metinleri = {normalle(metin[q]) for q, _, _ in test}
    ayni_metinli = {q for q, _, _ in train if normalle(metin[q]) in test_metinleri}

    # Kural 1 + 2: train'den çıkacak satırlar.
    train_cikan = [(q, d, s) for q, d, s in train if d in test_poz or q in ayni_metinli]
    cikan_kume = set(train_cikan)
    train_temiz = [r for r in train if r not in cikan_kume]

    # test_temiz: dokümanı train pozitifi olan satırlar çıkar. Sorgunun hiç satırı kalmazsa
    # o sorgu da kendiliğinden düşer (qrels'te yok = ölçülmez).
    test_cikan = [(q, d, s) for q, d, s in test if d in train_poz]
    test_cikan_kume = set(test_cikan)
    test_temiz = [r for r in test if r not in test_cikan_kume]

    return {
        "surum": SURUM,
        "ortak_dokuman_sayisi": len(ortak_dokuman),
        "ortak_dokumanlar": sorted(ortak_dokuman),
        "ayni_metinli_train_sorgulari": sorted(ayni_metinli),
        "sayilar": {
            "train_satir": len(train), "train_temiz_satir": len(train_temiz),
            "train_sorgu": len({q for q, _, _ in train}),
            "train_temiz_sorgu": len({q for q, _, _ in train_temiz}),
            "test_satir": len(test), "test_temiz_satir": len(test_temiz),
            "test_sorgu": len({q for q, _, _ in test}),
            "test_temiz_sorgu": len({q for q, _, _ in test_temiz}),
        },
        "_train_temiz": train_temiz,
        "_test_temiz": test_temiz,
        "train_cikarilan_qrels": [list(r[:2]) for r in sorted(train_cikan)],
        "test_cikarilan_qrels": [list(r[:2]) for r in sorted(test_cikan)],
    }


def tutarlilik_kontrolleri(sonuc: dict, girdi: Path) -> None:
    """Çıktı bozuksa sessizce yanlış sonuç vermesin: sınırları denetle."""
    train = qrels_oku(girdi / "qrels" / "train.tsv")
    test = qrels_oku(girdi / "qrels" / "test.tsv")
    test_poz = {d for _, d, s in test if s > 0}
    train_poz = {d for _, d, s in train if s > 0}
    tt, te = sonuc["_train_temiz"], sonuc["_test_temiz"]
    assert not any(d in test_poz for _, d, s in tt if s > 0), "temiz train'de test pozitifi kaldı"
    assert not any(d in train_poz for _, d, s in te if s > 0), "temiz test'te train pozitifi kaldı"
    assert set(tt) <= set(train) and set(te) <= set(test), "temiz küme orijinalin alt kümesi değil"
    assert len(tt) + len(sonuc["train_cikarilan_qrels"]) == len(train), "train satır sayıları tutmuyor"
    assert len(te) + len(sonuc["test_cikarilan_qrels"]) == len(test), "test satır sayıları tutmuyor"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SciFact sızıntı temizliği (Y-54).")
    ap.add_argument("--girdi", type=Path, default=VARSAYILAN_GIRDI, help="BEIR scifact klasörü")
    ap.add_argument("--cikti", type=Path, default=VARSAYILAN_CIKTI, help="temiz qrels klasörü")
    ap.add_argument("--liste", type=Path, default=LISTE_YOLU, help="çıkarılanların kayıt dosyası")
    ap.add_argument("--kontrol", action="store_true", help="hiçbir şey yazma; kayıt dosyasıyla karşılaştır")
    a = ap.parse_args(argv)

    for gerekli in ("qrels/train.tsv", "qrels/test.tsv", "queries.jsonl"):
        if not (a.girdi / gerekli).is_file():
            print(f"HATA: {a.girdi / gerekli} yok. Önce: python3 data/indir.py beir", file=sys.stderr)
            return 1

    try:
        sonuc = temizle(a.girdi)
        tutarlilik_kontrolleri(sonuc, a.girdi)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"HATA: girdi beklenen biçimde değil: {e}", file=sys.stderr)
        return 1
    kayit = {k: v for k, v in sonuc.items() if not k.startswith("_")}

    if a.kontrol:
        if not a.liste.is_file():
            print(f"HATA: {a.liste} yok", file=sys.stderr)
            return 1
        beklenen = json.loads(a.liste.read_text(encoding="utf-8"))
        if beklenen != kayit:
            print("FARKLI: bu makinedeki sonuç repodaki kayıtla aynı değil. Veriyi indir.py ile yeniden indir.", file=sys.stderr)
            return 1
        print(f"AYNI: {kayit['surum']} repodaki kayıtla birebir eşleşti.")
        return 0

    qrels_yaz(a.cikti / "train_temiz.tsv", sonuc["_train_temiz"])
    qrels_yaz(a.cikti / "test_temiz.tsv", sonuc["_test_temiz"])
    a.liste.parent.mkdir(parents=True, exist_ok=True)
    a.liste.write_text(json.dumps(kayit, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    s = kayit["sayilar"]
    print(f"ortak doküman: {kayit['ortak_dokuman_sayisi']}, aynı metinli train sorgusu: {len(kayit['ayni_metinli_train_sorgulari'])}")
    print(f"train: {s['train_satir']} satır / {s['train_sorgu']} sorgu -> {s['train_temiz_satir']} / {s['train_temiz_sorgu']}")
    print(f"test : {s['test_satir']} satır / {s['test_sorgu']} sorgu -> {s['test_temiz_satir']} / {s['test_temiz_sorgu']}")
    print(f"yazıldı: {a.cikti}/train_temiz.tsv, test_temiz.tsv ve {a.liste}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
