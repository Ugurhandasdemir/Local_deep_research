"""Model başına sorgu/doküman öneki (Y-55, ADIM 3a). İskelet: değerleri model kartından DOĞRULAYARAK doldur.

Neden var: bazı modeller sorguyu ve dokümanı farklı kalıpla görmek ister. Yanlış (ya da eksik) önek
modeli olduğundan kötü gösterir ve kıyas haksız olur (bkz. Y-03, Y-41).

Kural: değeri uydurma. Her modelin Hugging Face model kartındaki "usage" bölümünden oku, buraya
kaynak bağlantısıyla yaz. Emin olmadığın modelde None bırak ve onu çalıştırma.
"""

# "yontem" önekin NASIL uygulanacağını söyler:
#   "metin": sorgu/dokuman alanları elle eklenen metin öneki (None = önek yok).
#   "model": önek modelin kendi ayarında; kodla() encode_query/encode_document çağırır.
#            sorgu/dokuman burada None kalır, metni kütüphane ekler.
#   "task":  Jina tarzı; sorgu/dokuman, encode(task=...) için görev adıdır.
# "yukle" (isteğe bağlı): SentenceTransformer(...) çağrısına geçecek ek argümanlar.
ONEKLER = {
    "intfloat/multilingual-e5-base": {
        "yontem": "metin",
        "sorgu": "query: ",
        "dokuman": "passage: ",
        "kaynak": "https://huggingface.co/intfloat/multilingual-e5-base",
    },
    # Kart: sorguya talimat gerekmiyor, doküman öneksiz.
    "BAAI/bge-m3": {
        "yontem": "metin",
        "sorgu": None,
        "dokuman": None,
        "kaynak": "https://huggingface.co/BAAI/bge-m3",
    },
    # Kartın config_sentence_transformers.json dosyasındaki "query" prompt'u.
    # Talimat yalnız sorguya eklenir; "Query:" bitişik, satır sonu gerçek bir yeni satır.
    "Qwen/Qwen3-Embedding-0.6B": {
        "yontem": "metin",
        "sorgu": (
            "Instruct: Given a web search query, "
            "retrieve relevant passages that answer the query\nQuery:"
        ),
        "dokuman": None,
        "kaynak": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
    },
    # Kart örneği encode_query/encode_document kullanıyor. Kart girişli.
    "google/embeddinggemma-300m": {
        "yontem": "model",
        "sorgu": None,
        "dokuman": None,
        "kaynak": "https://huggingface.co/google/embeddinggemma-300m",
    },
    # Kart örneği prompt_name="query"/"document". Vision/audio yüklenmesin diye
    # config_kwargs gerekiyor. Kart girişli.
    "magibu/embeddingmagibu2": {
        "yontem": "model",
        "sorgu": None,
        "dokuman": None,
        "yukle": {"config_kwargs": {"vision_config": None, "audio_config": None}},
        "kaynak": "https://huggingface.co/magibu/embeddingmagibu2",
    },
    # Önek metni değil, encode(task=...) adaptörü. Metin prompt'ları da config'de
    # var, ikisinin birlikte mi kullanılacağı henüz doğrulanmadı.
    "jinaai/jina-embeddings-v3": {
        "yontem": "task",
        "sorgu": "retrieval.query",
        "dokuman": "retrieval.passage",
        "yukle": {"trust_remote_code": True},
        "kaynak": "https://huggingface.co/jinaai/jina-embeddings-v3",
    },
}


def _kayit_al(model_adi):
    """Modelin kaydını döner; yoksa anlaşılır bir hata verir (KeyError sızdırmaz)."""
    if model_adi not in ONEKLER:
        raise ValueError(
            f"{model_adi} için önek tanımlı değil, onekler.py'ye ekle "
            "(değeri model kartından doğrulayarak)."
        )
    return ONEKLER[model_adi]


def onek_al(model_adi):
    """-> (yontem, sorgu, dokuman)"""
    kayit = _kayit_al(model_adi)
    if kayit["yontem"] == "metin":
        # None = "kartı okudum, önek gerekmiyor". Bilerek boş metne çeviriyoruz.
        return "metin", kayit["sorgu"] or "", kayit["dokuman"] or ""
    # "model" ve "task" yöntemlerinde değerler olduğu gibi geçer.
    return kayit["yontem"], kayit["sorgu"], kayit["dokuman"]


def yukle_ayarlari(model_adi):
    """-> SentenceTransformer(...) için ek argümanlar (yoksa boş sözlük)."""
    return _kayit_al(model_adi).get("yukle", {})
