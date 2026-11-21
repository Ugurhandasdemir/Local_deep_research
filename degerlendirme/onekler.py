"""Model başına sorgu/doküman öneki (Y-55, ADIM 3a). Değerler model kartlarından okundu (2026-11-21).

Neden var: bazı modeller sorguyu ve dokümanı farklı kalıpla görmek ister. Yanlış (ya da eksik) önek
modeli olduğundan kötü gösterir ve kıyas haksız olur (bkz. Y-03, Y-41).

Kural: değeri uydurma. Her değer modelin kartından okundu, kaynak bağlantısı yanında. Ayrıntılı tablo:
data/MODELLER.md. Önek mantığıyla çalışmayan ya da doğrulanmamış modeller sözlükte YOK; onlar için
onek_al() hata verir (sessizce önek'siz çalışmasın diye).
"""

# Biçim: model adı -> {"sorgu": str | None, "dokuman": str | None, "kaynak": model kartı linki}
# None = "bu model önek istemiyor" demektir (kartta öyle yazıyor), bilgi eksikliği değil.
QWEN3_SORGU = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:"
)
QWEN3_KAYNAK = "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B"

ONEKLER = {
    "intfloat/multilingual-e5-base": {
        "sorgu": "query: ",
        "dokuman": "passage: ",
        "kaynak": "https://huggingface.co/intfloat/multilingual-e5-base",
    },
    # Qwen3: yalnız sorguya talimat eklenir, doküman olduğu gibi girer. Bu talimat kartın örneği;
    # görev başka bir şeyse (örn. bilimsel iddia doğrulama) cümleyi o göreve göre değiştirmek gerekebilir.
    "Qwen/Qwen3-Embedding-0.6B": {"sorgu": QWEN3_SORGU, "dokuman": None, "kaynak": QWEN3_KAYNAK},
    "Qwen/Qwen3-Embedding-4B": {"sorgu": QWEN3_SORGU, "dokuman": None, "kaynak": QWEN3_KAYNAK},
    "Qwen/Qwen3-Embedding-8B": {"sorgu": QWEN3_SORGU, "dokuman": None, "kaynak": QWEN3_KAYNAK},
    # BGE-M3: kart "sorgulara talimat eklemek artık gerekmiyor" diyor.
    "BAAI/bge-m3": {"sorgu": None, "dokuman": None, "kaynak": "https://huggingface.co/BAAI/bge-m3"},
    # Bilerek YOK (önek metniyle çalışmıyorlar, kodla_ara.py'de ayrı destek ister):
    #   jinaai/jina-embeddings-v3: önek yerine task="retrieval.query" / task="retrieval.passage" parametresi.
    #   google/embeddinggemma-2: önek yerine sentence-transformers istem adları ("SearchQuery", "Document").
    # BGE v1.5 gibi eski modellerin sorgu talimatı bu çalışmada okunmadı; gerekirse karttan doğrulayıp ekle.
}


def onek_al(model_adi):
    """-> (sorgu_oneki, dokuman_oneki)

    Model ONEKLER'de yoksa KeyError verir; sessizce önek'siz çalışmak kıyası bozar.
    Değeri None ise "" döndürür: bu, modelin önek istemediği anlamına gelir.
    """
    if model_adi not in ONEKLER:
        raise KeyError(f"'{model_adi}' için önek tanımlı değil, degerlendirme/onekler.py'ye ekle")
    kayit = ONEKLER[model_adi]
    return kayit["sorgu"] or "", kayit["dokuman"] or ""
