#!/usr/bin/env python3
"""Hazır veri setlerini indirir ve sağlamasını yapar (Y-40, tarif: data/VERI-TARIFI.md).

Kullanım:
    python3 data/indir.py --liste                 # setleri ve boyutlarını göster
    python3 data/indir.py beir                    # SciFact, NFCorpus, TREC-COVID
    python3 data/indir.py trmteb                  # SciFact-TR, NFCorpus-TR (huggingface_hub gerekir)
    python3 data/indir.py cohere                  # test + neighbors (küçük)
    python3 data/indir.py cohere --buyuk          # + shuffle_train (~3,1 GB)
    python3 data/indir.py uda                     # UDA kodu + paper_docs.zip (~0,85 GB)
    python3 data/indir.py hepsi --buyuk
    python3 data/indir.py beir --kuru             # hiçbir şey indirmeden planı göster

Veriler `data/raw/` altına iner (git bunu yok sayar, commit'lenmez). `--hedef` ile yer değişir.
Yalnızca standart kütüphane kullanır; tek istisna `trmteb` için `huggingface_hub`.
Sürüm, revision ve sağlama değerleri tariftekilerle aynı olmalı; birini değiştirirsen tarifin
yeni sürümünü de yaz (eski sürümü silme).
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

VARSAYILAN_HEDEF = Path(__file__).resolve().parent / "raw"

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
# ad -> sha256 (tarif §3)
BEIR = {
    "trec-covid": "120f42a7864d2214234537733c0d2c6684e42fdfafff2c5eacf98afca6656aa0",
    "nfcorpus": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
    "scifact": "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
}

# HF repo -> revision (tarif §4). MsMarco-TR bilerek yok (K-02).
TRMTEB = {
    "trmteb/scifact-tr": "e86dcd706c591848eed192fe6376a9ad4675854d",
    "trmteb/nfcorpus-tr": "c7e02f8114f4ca4ef9c2d47df193415928ef992e",
}

COHERE_URL = "https://assets.zilliz.com/benchmark/cohere_medium_1m"
# dosya -> (beklenen boyut bayt, büyük mü). sha256 yayımlanmıyor, bu yüzden boyut kontrol edilir
# ve hesaplanan sha256 yazdırılır; tarife kaydedilmesi için (tarif §5).
COHERE = {
    "test.parquet": (3_133_165, False),
    "neighbors.parquet": (3_704_127, False),
    "shuffle_train.parquet": (3_134_418_448, True),
}

UDA_REPO = "https://github.com/qinchuanhui/UDA-Benchmark"
UDA_COMMIT = "fca5237ac316e776d8dbccffa55ca29c0efdc185"
UDA_HF_REV = "d4367103fe8fe86b3bb76c66be8eafc4fb4117b2"
UDA_ZIP_URL = (
    "https://huggingface.co/datasets/qinchuanhui/UDA-QA/resolve/"
    f"{UDA_HF_REV}/src_doc_files/paper_docs.zip"
)
UDA_ZIP_SHA256 = "74890c04b6956c1a2f04adcd7946c0b9cc133a647ccc68854d6e9b5771b7a66a"


class IndirmeHatasi(Exception):
    """Sağlama tutmadı, boyut yanlış ya da ağ hatası."""


def sha256_dosya(yol: Path) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def indir_dosya(url: str, hedef: Path, sha256: str = None, boyut: int = None, kuru: bool = False) -> str:
    """Dosyayı indirir, sağlamasını yapar. Zaten doğruysa tekrar indirmez. Hesaplanan sha256'yı döner."""
    if kuru:
        print(f"  [kuru] {url} -> {hedef}")
        return ""
    hedef.parent.mkdir(parents=True, exist_ok=True)

    if hedef.exists() and (boyut is None or hedef.stat().st_size == boyut):
        ozet = sha256_dosya(hedef)
        if sha256 is None or ozet == sha256:
            print(f"  var, doğrulandı: {hedef.name}")
            return ozet
        print(f"  var ama sağlama tutmuyor, yeniden indiriliyor: {hedef.name}")
        hedef.unlink()

    # Yarım kalan indirme .part dosyasında durur ve Range ile devam eder.
    parca = hedef.with_name(hedef.name + ".part")
    basla = parca.stat().st_size if parca.exists() else 0
    istek = urllib.request.Request(url, headers={"User-Agent": "ldr-indir/1"})
    if basla:
        istek.add_header("Range", f"bytes={basla}-")
    try:
        with urllib.request.urlopen(istek, timeout=60) as yanit:
            if basla and yanit.status != 206:
                basla = 0  # sunucu Range desteklemedi, baştan başla
            mod = "ab" if basla else "wb"
            with open(parca, mod) as f:
                shutil.copyfileobj(yanit, f, 1 << 20)
    except OSError as e:  # URLError ve ağ zaman aşımı OSError'dan türer
        raise IndirmeHatasi(f"{url}: {e}") from e

    if boyut is not None and parca.stat().st_size != boyut:
        gercek = parca.stat().st_size
        if gercek > boyut:  # bozuk içerik, devam edilemez
            parca.unlink()
            raise IndirmeHatasi(f"{hedef.name}: boyut {gercek} bayt, beklenen {boyut}. Dosya silindi.")
        raise IndirmeHatasi(f"{hedef.name}: eksik ({gercek}/{boyut} bayt). Yeniden çalıştırınca devam eder.")
    ozet = sha256_dosya(parca)
    if sha256 is not None and ozet != sha256:
        parca.unlink()
        raise IndirmeHatasi(f"{hedef.name}: sha256 tutmuyor (bulunan {ozet}, beklenen {sha256}). Dosya silindi.")
    parca.replace(hedef)
    print(f"  indirildi: {hedef.name} ({hedef.stat().st_size} bayt) sha256={ozet}")
    return ozet


def guvenli_ac(zip_yolu: Path, hedef: Path) -> None:
    """Zip'i açar; hedef klasörün dışına yazmaya çalışan üyeyi reddeder (indirilen veri güvenilmez)."""
    kok = hedef.resolve()
    with zipfile.ZipFile(zip_yolu) as z:
        for uye in z.namelist():
            if not (kok / uye).resolve().is_relative_to(kok):  # Python 3.9+
                raise IndirmeHatasi(f"{zip_yolu.name}: güvensiz yol: {uye}")
        z.extractall(hedef)


def beir(hedef: Path, kuru: bool, **_):
    print("BEIR (SciFact, NFCorpus, TREC-COVID)")
    for ad, sha in BEIR.items():
        zip_yolu = hedef / f"{ad}.zip"
        indir_dosya(f"{BEIR_URL}/{ad}.zip", zip_yolu, sha256=sha, kuru=kuru)
        if not kuru and not (hedef / ad).is_dir():
            guvenli_ac(zip_yolu, hedef)
            print(f"  açıldı: {hedef / ad}")


def trmteb(hedef: Path, kuru: bool, **_):
    print("TR-MTEB (SciFact-TR, NFCorpus-TR)")
    for repo, rev in TRMTEB.items():
        yer = hedef / repo.split("/")[1]
        if kuru:
            print(f"  [kuru] {repo}@{rev} -> {yer}")
            continue
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise IndirmeHatasi("huggingface_hub yok: pip install 'huggingface_hub>=0.23'")
        snapshot_download(repo_id=repo, repo_type="dataset", revision=rev, local_dir=str(yer))
        print(f"  indirildi: {repo}@{rev[:8]}")


def cohere(hedef: Path, kuru: bool, buyuk: bool = False, **_):
    print("VectorDBBench Cohere 1M (yalnız DB testi)")
    yer = hedef / "cohere_medium_1m"
    for ad, (boyut, buyuk_mu) in COHERE.items():
        if buyuk_mu and not buyuk:
            print(f"  atlandı (büyük, ~{boyut / 1e9:.1f} GB): {ad}  -> --buyuk ile indir")
            continue
        indir_dosya(f"{COHERE_URL}/{ad}", yer / ad, boyut=boyut, kuru=kuru)


def uda(hedef: Path, kuru: bool, **_):
    print("UDA (PaperTab + PaperText)")
    repo_yolu = hedef / "UDA-Benchmark"
    if kuru:
        print(f"  [kuru] git clone {UDA_REPO} @ {UDA_COMMIT[:8]} -> {repo_yolu}")
    elif not repo_yolu.is_dir():
        hedef.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", UDA_REPO, str(repo_yolu)], check=True)
        subprocess.run(["git", "-C", str(repo_yolu), "checkout", "-q", UDA_COMMIT], check=True)
        print(f"  kod hazır: {UDA_COMMIT[:8]}")
    else:
        print("  kod klasörü var, atlandı")
    zip_yolu = hedef / "paper_docs.zip"
    indir_dosya(UDA_ZIP_URL, zip_yolu, sha256=UDA_ZIP_SHA256, kuru=kuru)
    if not kuru and not (hedef / "paper_docs").is_dir():
        guvenli_ac(zip_yolu, hedef)
        print(f"  açıldı: {hedef / 'paper_docs'}")
    print("  UYARI: PDF'lerin telifi karışık, repoya ya da vault'a koyma (tarif §6).")


SETLER = {"beir": beir, "trmteb": trmteb, "cohere": cohere, "uda": uda}


def liste():
    print("beir     ~80 MB    SciFact, NFCorpus, TREC-COVID (sha256 doğrulanır)")
    print("trmteb   ~10 MB    SciFact-TR, NFCorpus-TR (HF revision sabit, huggingface_hub gerekir)")
    print("cohere   ~7 MB     test + neighbors; --buyuk ile +3,1 GB shuffle_train")
    print("uda      ~0,85 GB  UDA kodu (commit sabit) + paper_docs.zip (sha256 doğrulanır)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hazır veri setlerini indir (Y-40).")
    ap.add_argument("set", nargs="?", choices=[*SETLER, "hepsi"], help="indirilecek set")
    ap.add_argument("--liste", action="store_true", help="setleri göster ve çık")
    ap.add_argument("--kuru", action="store_true", help="indirmeden planı göster")
    ap.add_argument("--buyuk", action="store_true", help="3 GB üstü dosyaları da indir (Cohere shuffle_train)")
    ap.add_argument("--hedef", type=Path, default=VARSAYILAN_HEDEF, help="indirme klasörü (varsayılan data/raw)")
    a = ap.parse_args(argv)

    if a.liste:
        liste()
        return 0
    if not a.set:
        ap.error("bir set adı ver (ya da --liste)")
    secilen = list(SETLER) if a.set == "hepsi" else [a.set]
    try:
        for ad in secilen:
            SETLER[ad](hedef=a.hedef, kuru=a.kuru, buyuk=a.buyuk)
    except (IndirmeHatasi, subprocess.CalledProcessError) as e:
        print(f"HATA: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
