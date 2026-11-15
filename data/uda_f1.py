#!/usr/bin/env python3
"""Loads the Y-57 question subset and scores answers with token F1 (spec: data/VERI-TARIFI.md section 6).
Y-57 soru alt kümesini yükler ve cevapları token F1 ile puanlar (tarif: data/VERI-TARIFI.md bölüm 6).

We take ONLY token_f1_score from UDA's basic_utils.py. We never import my_eval.py, because its
finance path runs model output with exec().
UDA'nın basic_utils.py dosyasından YALNIZ token_f1_score alınır. my_eval.py hiç içe aktarılmaz,
çünkü finans yolu model çıktısını exec() ile çalıştırır.

Usage / Kullanım:
    python3 data/uda_altkume.py     # first make the subset list / önce alt küme listesini üret
    python3 data/uda_f1.py          # prints a short summary of the subset / alt kümenin özetini yazar

From other code / Başka koddan:
    from uda_f1 import altkume_sorulari, soru_f1
"""
import csv
import importlib.util
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent
UDA = DATA / "raw" / "UDA-Benchmark"
LISTE_YOLU = DATA / "listeler" / "uda_altkume_v1.json"
DOSYALAR = {"paper_tab": "paper_tab_qa.csv", "paper_text": "paper_text_qa.csv"}


def token_f1_yukle():
    """Loads token_f1_score straight from the pinned UDA file and returns it.
    token_f1_score fonksiyonunu sabitlenmiş UDA dosyasından doğrudan yükleyip döner.

    Why not a normal import: the file lives in a cloned folder, not in an installed package.
    importlib lets us load a module from an exact file path.
    Neden normal import değil: dosya kurulu bir paketin değil, klonlanmış bir klasörün içinde.
    importlib, modülü tam bir dosya yolundan yüklememizi sağlar.
    """
    yol = UDA / "uda" / "eval" / "utils" / "basic_utils.py"
    belirtim = importlib.util.spec_from_file_location("uda_basic_utils", yol)
    modul = importlib.util.module_from_spec(belirtim)
    belirtim.loader.exec_module(modul)  # runs the file once so its functions exist / fonksiyonlar oluşsun diye dosyayı bir kez çalıştırır
    return modul.token_f1_score


token_f1_score = token_f1_yukle()


def soru_f1(tahmin: str, referanslar: list) -> float:
    """Best token F1 of one prediction against all reference answers.
    Bir tahminin tüm referans cevaplara karşı en iyi token F1 değeri.

    Empty references are skipped (many questions have only one answer). With no usable
    reference the question cannot be scored, so we raise instead of returning 0.
    Boş referanslar atlanır (çok sorunun tek cevabı var). Kullanılabilir referans yoksa soru
    puanlanamaz; 0 döndürmek yerine hata veriyoruz.
    """
    # List comprehension with a filter: keep only references that are not blank.
    # Süzgeçli liste üreteci: yalnız boş olmayan referansları tutar.
    dolu = [r for r in referanslar if r.strip()]
    if not dolu:
        raise ValueError("kullanılabilir referans cevap yok")
    # max() over a generator: score against each reference, take the highest.
    # Üreteç üzerinde max(): her referansla puanla, en yükseğini al.
    return float(max(token_f1_score(tahmin, r) for r in dolu))


def altkume_sorulari() -> list:
    """Returns the subset as a list of dicts: set, q_uid, doc_name, question, references.
    Alt kümeyi sözlük listesi olarak döner: set, q_uid, doc_name, question, references.

    The order is fixed (sorted by set name, then by the saved q_uid order).
    Sıra sabittir (set adına göre, sonra kayıtlı q_uid sırasına göre).
    """
    with open(LISTE_YOLU, encoding="utf-8") as f:
        kayit = json.load(f)
    sonuc = []
    for ad in sorted(kayit["q_uid"]):
        istenen = set(kayit["q_uid"][ad])  # set: fast "is it in the list?" check / hızlı "listede mi?" kontrolü
        with open(UDA / "dataset" / "qa" / DOSYALAR[ad], encoding="utf-8", newline="") as f:
            satirlar = list(csv.reader(f, delimiter="|"))
        # Index the CSV rows by q_uid so we can pick them in the saved order.
        # CSV satırlarını q_uid ile dizinleyip kayıtlı sırayla seçiyoruz.
        satir_ile = {s[1]: s for s in satirlar[1:] if s[1] in istenen}
        for q_uid in kayit["q_uid"][ad]:
            s = satir_ile[q_uid]  # KeyError here means the CSV and the list do not match / KeyError: CSV ile liste uyuşmuyor
            sonuc.append(
                {
                    "set": ad,
                    "q_uid": q_uid,
                    "doc_name": s[0],
                    "question": s[2],
                    "references": s[3:6],
                }
            )
    return sonuc


def main() -> None:
    sorular = altkume_sorulari()
    print(f"alt küme: {len(sorular)} soru")
    for ad in sorted(DOSYALAR):
        alt = [s for s in sorular if s["set"] == ad]
        pdf = len({s["doc_name"] for s in alt})
        print(f"  {ad}: {len(alt)} soru, {pdf} farklı PDF")
    # Sanity check: a reference scored against itself must give exactly 1.0.
    # Sağlama: bir referansın kendisiyle puanı tam 1.0 olmalı.
    ilk = sorular[0]
    print("kendisiyle F1:", soru_f1(ilk["references"][0], ilk["references"]))
    print("boş tahmin F1:", soru_f1("", ilk["references"]))


if __name__ == "__main__":
    main()
