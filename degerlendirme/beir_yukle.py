"""BEIR setini diskten okur (Y-55, ADIM 2).

Beklenen klasör (data/indir.py beir çıktısı):
    <klasor>/corpus.jsonl    her satır: {"_id": "...", "title": "...", "text": "..."}
    <klasor>/queries.jsonl   her satır: {"_id": "...", "text": "..."}
    <klasor>/qrels/<bolum>.tsv   başlık: query-id<TAB>corpus-id<TAB>score

Temizlenmiş qrels için (Y-54) `qrels_yolu` ile data/raw/scifact_temiz/test_temiz.tsv verilir.

Bu dosyanın ürettiği üç veri yapısı (sonraki dosyalar bunları kullanır):

    corpus   : {"D1": {"title": "...", "text": "..."}, "D2": {...}, ...}
    sorgular : {"q1": "sorgu metni", "q2": "...", ...}
    qrels    : {"q1": {"D1", "D5"}, "q2": {"D3"}, ...}      <- değerler KÜME (set)

Neden sözlük (dict)? Id ile doğrudan erişim hızlı: corpus["D1"]. Liste olsaydı her seferinde taranırdı.
Neden qrels değerleri küme? metrikler.py'deki `dogrular` parametresi zaten set bekliyor ve
`d in kume` kontrolü küme/sözlükte çok hızlıdır.
"""
import csv
import json
from pathlib import Path


def _jsonl_satirlari(yol):
    """Bir .jsonl dosyasını satır satır okuyup her satırı sözlüğe çevirerek verir.

    JSONL = "JSON Lines": her satır bağımsız bir JSON nesnesi. Dosyayı tek seferde belleğe
    yüklemek yerine satır satır işleriz (TREC-COVID'in corpus'u yüzlerce MB).

    `yield` kullanan fonksiyon bir "üreteç"tir (generator): her çağrıda bir sonraki satırı verir,
    hepsini listede toplamaz. `for o in _jsonl_satirlari(...)` ile dolaşılır.
    """
    yol = Path(yol)
    if not yol.is_file():
        raise FileNotFoundError(f"{yol} yok. Önce: python3 data/indir.py beir")
    # `with open(...)` dosyayı iş bitince (hata olsa bile) kapatır. encoding'i açıkça yazıyoruz,
    # çünkü varsayılan işletim sistemine göre değişir ve Türkçe karakterleri bozabilir.
    with open(yol, encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if satir:  # boş satırları atla
                yield json.loads(satir)


def corpus_oku(klasor):
    """-> dict[str, dict]:  id -> {"title": str, "text": str}

    Boş "text" olabilir (TREC-COVID'de %24,6!). Bu hata değil, o dokümanı atmıyoruz: başlıkla aranır.
    """
    corpus = {}
    for o in _jsonl_satirlari(Path(klasor) / "corpus.jsonl"):
        # .get(anahtar, varsayılan): anahtar yoksa hata vermez, varsayılanı döndürür.
        corpus[o["_id"]] = {
            "title": o.get("title", ""),
            "text": o.get("text", ""),
        }
    return corpus


def sorgular_oku(klasor):
    """-> dict[str, str]:  sorgu id -> sorgu metni"""
    sorgular = {}
    for o in _jsonl_satirlari(Path(klasor) / "queries.jsonl"):
        sorgular[o["_id"]] = o["text"]
    return sorgular


def qrels_oku(yol):
    """-> dict[str, set[str]]:  sorgu id -> DOĞRU doküman id'leri kümesi (skor > 0 olanlar)

    Skor 0 "ilgisiz" demektir (bazı setlerde açıkça işaretlenir), bunları doğru saymıyoruz.
    """
    yol = Path(yol)
    if not yol.is_file():
        raise FileNotFoundError(f"{yol} yok")
    qrels = {}
    with open(yol, encoding="utf-8", newline="") as f:
        okuyucu = csv.reader(f, delimiter="\t")  # TSV = sekmeyle ayrılmış değerler
        baslik = next(okuyucu)  # ilk satır: başlık. next() bir sonraki satırı alır ve ilerler.
        if baslik != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"{yol}: beklenen başlık yok, bulunan: {baslik}")
        for sorgu_id, dokuman_id, skor in okuyucu:
            if int(skor) > 0:
                # setdefault(anahtar, varsayılan): anahtar yoksa varsayılanı (burada boş küme)
                # ekler ve DÖNDÜRÜR; varsa mevcut değeri döndürür. Böylece "var mı?" kontrolü gerekmez.
                qrels.setdefault(sorgu_id, set()).add(dokuman_id)
    return qrels


def yukle(klasor, bolum="test", qrels_yolu=None):
    """-> (corpus, sorgular, qrels)

    Yalnız qrels'te adı geçen sorgular ölçülür: sorgular, qrels'in anahtarlarına göre süzülür.
    (queries.jsonl tüm bölümlerin sorgularını içerir; test ölçerken train sorgularını almayız.)
    """
    klasor = Path(klasor)
    if qrels_yolu is None:
        qrels_yolu = klasor / "qrels" / f"{bolum}.tsv"
    corpus = corpus_oku(klasor)
    tum_sorgular = sorgular_oku(klasor)
    qrels = qrels_oku(qrels_yolu)

    # Önce tutarlılık: qrels'teki her sorgunun metni queries.jsonl'de olmalı.
    eksik = [q for q in qrels if q not in tum_sorgular]
    if eksik:
        raise ValueError(f"qrels'te olup queries.jsonl'de olmayan {len(eksik)} sorgu var, örn. {eksik[:3]}")
    # Sözlük üreteci (dict comprehension): yeni sözlük = {anahtar: değer for ... in ...}
    sorgular = {q: tum_sorgular[q] for q in qrels}
    return corpus, sorgular, qrels
