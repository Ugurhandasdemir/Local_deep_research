#!/usr/bin/env python3
"""Picks the fixed UDA question subset (Y-57, spec: data/VERI-TARIFI.md section 6).
UDA'dan sabit soru alt kümesini seçer (Y-57, tarif: data/VERI-TARIFI.md bölüm 6).

What it does / Ne yapar:
    PaperTab (393 questions) and PaperText (2804 questions) are sampled proportionally to a
    total of about 300 questions, with seed 42. Only the question ids (q_uid) are saved, so all
    three of us evaluate exactly the same questions.
    PaperTab (393 soru) ve PaperText (2804 soru) toplam yaklaşık 300 soruya orantılı örneklenir,
    seed 42. Yalnız soru kimlikleri (q_uid) kaydedilir; üçümüz de aynı sorularla ölçeriz.

Usage / Kullanım:
    python3 data/indir.py uda            # or only: git clone UDA-Benchmark into data/raw
    python3 data/uda_altkume.py          # writes data/listeler/uda_altkume_v1.json

Standard library only. The output is deterministic: same seed, same list.
Yalnızca standart kütüphane. Çıktı deterministiktir: aynı seed, aynı liste.
"""
import csv
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent
SORU_KLASORU = DATA / "raw" / "UDA-Benchmark" / "dataset" / "qa"
CIKTI = DATA / "listeler" / "uda_altkume_v1.json"
# If the rule changes, open v2 and do not delete v1 (spec rule).
# Kural değişirse v2 aç, v1'i silme (tarif kuralı).
SURUM = "uda_altkume_v1"
UDA_COMMIT = "fca5237ac316e776d8dbccffa55ca29c0efdc185"  # same as data/indir.py / indir.py ile aynı
TOHUM = 42
TOPLAM = 300
DOSYALAR = {"paper_tab": "paper_tab_qa.csv", "paper_text": "paper_text_qa.csv"}


def sorulari_oku(yol: Path):
    """Returns the list of q_uid values in one UDA question file.
    Bir UDA soru dosyasındaki q_uid değerlerinin listesini döner.

    The file uses "|" instead of a comma, so we pass delimiter="|".
    Dosya virgül yerine "|" kullanır, bu yüzden delimiter="|" veriyoruz.
    """
    # "with open(...) as f" closes the file automatically when the block ends.
    # "with open(...) as f" blok bitince dosyayı kendiliğinden kapatır.
    with open(yol, encoding="utf-8", newline="") as f:
        satirlar = list(csv.reader(f, delimiter="|"))
    if satirlar[0] != ["doc_name", "q_uid", "question", "answer_1", "answer_2", "answer_3"]:
        raise ValueError(f"{yol}: beklenen başlık yok")
    # List comprehension: build a new list from every row except the header; column 1 is q_uid.
    # Liste üreteci: başlık hariç her satırdan yeni bir liste kurar; 1. sütun q_uid.
    return [satir[1] for satir in satirlar[1:]]


def kac_soru(sayilar: dict, toplam: int) -> dict:
    """Splits `toplam` between the files in proportion to their sizes.
    `toplam` sayısını dosyalara büyüklükleriyle orantılı böler.

    Example / Örnek: {"paper_tab": 393, "paper_text": 2804}, 300 -> {"paper_tab": 37, "paper_text": 263}
    The last file takes the remainder, so the sum is always exactly `toplam`.
    Son dosya kalanı alır, böylece toplam her zaman tam `toplam` olur.
    """
    adlar = sorted(sayilar)
    genel = sum(sayilar.values())
    sonuc = {}
    for ad in adlar[:-1]:
        sonuc[ad] = round(toplam * sayilar[ad] / genel)
    sonuc[adlar[-1]] = toplam - sum(sonuc.values())
    return sonuc


def main() -> None:
    # dict comprehension: {file name: [all q_uids]} / sözlük üreteci: {ad: [tüm q_uid'ler]}
    tum = {ad: sorulari_oku(SORU_KLASORU / dosya) for ad, dosya in DOSYALAR.items()}
    sayilar = {ad: len(idler) for ad, idler in tum.items()}
    adetler = kac_soru(sayilar, TOPLAM)

    # A private Random object keeps our seed separate from any other code that uses random.
    # Özel bir Random nesnesi, tohumumuzu random kullanan diğer kodlardan ayrı tutar.
    rastgele = random.Random(TOHUM)
    secilen = {}
    for ad in sorted(tum):
        # sorted() first: the pick must not depend on the order in the CSV file.
        # Önce sorted(): seçim CSV dosyasındaki sıraya bağlı olmasın.
        secilen[ad] = sorted(rastgele.sample(sorted(tum[ad]), adetler[ad]))

    kayit = {
        "surum": SURUM,
        "tohum": TOHUM,
        "uda_commit": UDA_COMMIT,
        "dagilim": "orantili",
        "toplam_soru": {ad: sayilar[ad] for ad in sorted(sayilar)},
        "secilen_adet": adetler,
        "q_uid": secilen,
    }
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    with open(CIKTI, "w", encoding="utf-8", newline="") as f:
        # sort_keys and indent make the file stable, so git diff stays readable.
        # sort_keys ve indent dosyayı kararlı yapar, git diff okunur kalır.
        json.dump(kayit, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"yazıldı: {CIKTI}")
    for ad in sorted(adetler):
        print(f"  {ad}: {adetler[ad]} / {sayilar[ad]}")


if __name__ == "__main__":
    main()
