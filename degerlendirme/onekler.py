"""Model başına sorgu/doküman öneki (Y-55, ADIM 3a). İskelet: değerleri model kartından DOĞRULAYARAK doldur.

Neden var: bazı modeller sorguyu ve dokümanı farklı kalıpla görmek ister. Yanlış (ya da eksik) önek
modeli olduğundan kötü gösterir ve kıyas haksız olur (bkz. Y-03, Y-41).

Kural: değeri uydurma. Her modelin Hugging Face model kartındaki "usage" bölümünden oku, buraya
kaynak bağlantısıyla yaz. Emin olmadığın modelde None bırak ve onu çalıştırma.
"""

# Biçim: model adı -> {"sorgu": str | None, "dokuman": str | None, "kaynak": model kartı linki}
# Qwen3 gibi "talimat" isteyen modellerde sorgu öneki bir cümle olabilir.
ONEKLER = {
    # TODO: "intfloat/multilingual-e5-base": {"sorgu": ..., "dokuman": ..., "kaynak": "https://huggingface.co/..."},
    # TODO: "Qwen/Qwen3-Embedding-0.6B": {...},
    # TODO: "BAAI/bge-m3": {...},
    # TODO: EmbeddingGemma, Jina Embeddings v3 (Y-41 listesi)
}


def onek_al(model_adi):
    """-> (sorgu_oneki, dokuman_oneki)

    TODO: model ONEKLER'de yoksa KeyError yerine anlaşılır bir hata ver
          ("bu model için önek tanımlı değil, onekler.py'ye ekle").
          Değeri None ise "" döndür (önek yok demektir) ama bunu bilerek yap.
    """
    raise NotImplementedError
