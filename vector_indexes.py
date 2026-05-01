import hashlib
import os
import uuid
from functools import lru_cache
from typing import Any


BASE_DIR = os.path.dirname(__file__)
DB_DIR = os.path.join(BASE_DIR, "DB")
MODELS_DIR = os.path.join(BASE_DIR, "models")

WRITE_BATCH = 32
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

SCENARIOS = {
    "fast": {"db": "chromadb", "embedding": "all_mini_l6"},
    "balanced": {"db": "weaviate", "embedding": "bge_squad"},
    "best": {"db": "milvus", "embedding": "bge_squad"},
}

MODEL_CONFIGS = {
    "all_mini_l6": {
        "path": os.path.join(MODELS_DIR, "all_mini_l6_v2"),
        "dim": 384,
    },
    "bge_squad": {
        "path": os.path.join(MODELS_DIR, "bge_squad_model"),
        "dim": 1024,
    },
}


def resolve_config(
    scenario: str | None = None,
    db: str | None = None,
    embedding: str | None = None,
) -> dict[str, str]:
    scenario_key = (scenario or "balanced").strip()
    cfg = dict(SCENARIOS.get(scenario_key, SCENARIOS["balanced"]))

    if db:
        cfg["db"] = db.strip().lower()
    if embedding:
        cfg["embedding"] = embedding.strip()

    if cfg["db"] not in {"chromadb", "weaviate", "milvus"}:
        raise ValueError(f"Desteklenmeyen DB seçimi: {cfg['db']}")
    if cfg["embedding"] not in MODEL_CONFIGS:
        raise ValueError(f"Desteklenmeyen embedding modeli: {cfg['embedding']}")
    return cfg


def scenario_targets() -> list[dict[str, str]]:
    targets = []
    seen = set()
    for name, cfg in SCENARIOS.items():
        key = (cfg["db"], cfg["embedding"])
        if key in seen:
            continue
        seen.add(key)
        targets.append({"scenario": name, **cfg})
    return targets


def chunk_text(text: str) -> list[str]:
    clean = " ".join((text or "").split())
    if not clean:
        return []

    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    chunks = []
    for i in range(0, len(clean), step):
        chunk = clean[i:i + CHUNK_SIZE].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks


@lru_cache(maxsize=4)
def _embedding_model(model_key: str):
    from sentence_transformers import SentenceTransformer

    cfg = MODEL_CONFIGS[model_key]
    return SentenceTransformer(cfg["path"], device="cpu")


def _encode(model_key: str, chunks: list[str]) -> list[list[float]]:
    model = _embedding_model(model_key)
    embeddings = model.encode(
        chunks,
        batch_size=WRITE_BATCH,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [emb.tolist() for emb in embeddings]


def _stable_id(prefix: str, model_key: str, filename: str, chunk_idx: int) -> str:
    return f"{prefix}:{model_key}:{filename}:{chunk_idx}"


def _stable_int_id(model_key: str, filename: str, chunk_idx: int) -> int:
    raw = f"{model_key}:{filename}:{chunk_idx}".encode("utf-8")
    digest = hashlib.sha1(raw).digest()
    return 1_000_000_000_000 + (int.from_bytes(digest[:8], "big") % 8_000_000_000_000_000_000)


def _stable_uuid(model_key: str, filename: str, chunk_idx: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{model_key}:{filename}:{chunk_idx}"))


def _batched(items: list[Any], size: int):
    for i in range(0, len(items), size):
        yield i, items[i:i + size]


def index_document_for_all_scenarios(filename: str, full_text: str) -> list[dict[str, Any]]:
    chunks = chunk_text(full_text)
    if not chunks:
        return [{"scenario": "all", "status": "skipped", "message": "Metin parçası üretilemedi"}]

    by_model: dict[str, list[list[float]]] = {}
    results = []
    for target in scenario_targets():
        model_key = target["embedding"]
        if model_key not in by_model:
            by_model[model_key] = _encode(model_key, chunks)

        try:
            count = index_document_for_config(filename, chunks, by_model[model_key], target)
            results.append({**target, "status": "success", "chunks": count})
        except Exception as exc:
            results.append({**target, "status": "error", "message": str(exc)})
    return results


def index_document_for_config(
    filename: str,
    chunks: list[str],
    embeddings: list[list[float]],
    cfg: dict[str, str],
) -> int:
    db = cfg["db"]
    model_key = cfg["embedding"]
    if db == "chromadb":
        return _index_chromadb(model_key, filename, chunks, embeddings)
    if db == "milvus":
        return _index_milvus(model_key, filename, chunks, embeddings)
    if db == "weaviate":
        return _index_weaviate(model_key, filename, chunks, embeddings)
    raise ValueError(f"Desteklenmeyen DB: {db}")


def _index_chromadb(model_key: str, filename: str, chunks: list[str], embeddings: list[list[float]]) -> int:
    import chromadb

    collection = f"docs_{model_key}"
    db_path = os.path.join(DB_DIR, "chromadb", f"{model_key}_db")
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    for start, batch_chunks in _batched(chunks, WRITE_BATCH):
        end = start + len(batch_chunks)
        ids = [_stable_id("upload", model_key, filename, idx) for idx in range(start, end)]
        col.upsert(
            ids=ids,
            embeddings=embeddings[start:end],
            documents=batch_chunks,
            metadatas=[
                {"source": filename, "chunk": idx, "origin": "upload"}
                for idx in range(start, end)
            ],
        )
    return len(chunks)


def _index_milvus(model_key: str, filename: str, chunks: list[str], embeddings: list[list[float]]) -> int:
    from pymilvus import MilvusClient

    collection = f"docs_{model_key}"
    db_path = os.path.join(DB_DIR, "milvus", f"{model_key}_db.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    client = MilvusClient(db_path)

    if collection not in client.list_collections():
        client.create_collection(
            collection_name=collection,
            dimension=MODEL_CONFIGS[model_key]["dim"],
            metric_type="COSINE",
        )

    for start, batch_chunks in _batched(chunks, WRITE_BATCH):
        rows = []
        for local_idx, chunk in enumerate(batch_chunks):
            idx = start + local_idx
            rows.append(
                {
                    "id": _stable_int_id(model_key, filename, idx),
                    "vector": embeddings[idx],
                    "text": chunk,
                    "source": filename,
                    "chunk": idx,
                    "origin": "upload",
                }
            )
        client.upsert(collection_name=collection, data=rows)
    return len(chunks)


def _ensure_weaviate_collection(client, collection_name: str):
    from weaviate.classes.config import Configure, DataType, Property

    if client.collections.exists(collection_name):
        return client.collections.get(collection_name)

    client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
        ],
    )
    return client.collections.get(collection_name)


def _index_weaviate(model_key: str, filename: str, chunks: list[str], embeddings: list[list[float]]) -> int:
    import weaviate

    collection_name = f"Docs_{model_key}"
    client = weaviate.connect_to_local()
    try:
        col = _ensure_weaviate_collection(client, collection_name)
        for idx, chunk in enumerate(chunks):
            uid = _stable_uuid(model_key, filename, idx)
            properties = {"text": chunk, "source": filename}
            if col.data.exists(uid):
                col.data.replace(uuid=uid, properties=properties, vector=embeddings[idx])
            else:
                col.data.insert(properties=properties, vector=embeddings[idx], uuid=uid)
        return len(chunks)
    finally:
        client.close()


def search_vector_index(
    query: str,
    scenario: str | None = None,
    db: str | None = None,
    embedding: str | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    cfg = resolve_config(scenario=scenario, db=db, embedding=embedding)
    model_key = cfg["embedding"]
    vector = _encode(model_key, [query])[0]

    if cfg["db"] == "chromadb":
        return _search_chromadb(model_key, vector, top_k)
    if cfg["db"] == "milvus":
        return _search_milvus(model_key, vector, top_k)
    if cfg["db"] == "weaviate":
        return _search_weaviate(model_key, vector, top_k)
    raise ValueError(f"Desteklenmeyen DB: {cfg['db']}")


def _source_url_result(rank: int, filename: str, score: float, text: str) -> dict[str, Any]:
    return {
        "sirasi": rank,
        "dosya_adi": filename,
        "benzerlik": float(f"{score:.4f}"),
        "metin": text,
    }


def _search_chromadb(model_key: str, vector: list[float], top_k: int) -> list[dict[str, Any]]:
    import chromadb

    collection = f"docs_{model_key}"
    db_path = os.path.join(DB_DIR, "chromadb", f"{model_key}_db")
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_collection(collection)
    results = col.query(
        query_embeddings=[vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    out = []
    for idx, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
        1,
    ):
        out.append(_source_url_result(idx, meta.get("source") or meta.get("filename") or "", 1 - dist, doc))
    return out


def _search_milvus(model_key: str, vector: list[float], top_k: int) -> list[dict[str, Any]]:
    from pymilvus import MilvusClient

    collection = f"docs_{model_key}"
    db_path = os.path.join(DB_DIR, "milvus", f"{model_key}_db.db")
    client = MilvusClient(db_path)
    results = client.search(
        collection_name=collection,
        data=[vector],
        limit=top_k,
        output_fields=["text", "source"],
    )

    out = []
    for idx, hit in enumerate(results[0], 1):
        entity = hit.get("entity", {})
        out.append(
            _source_url_result(
                idx,
                entity.get("source") or "",
                hit.get("distance", 0.0),
                entity.get("text") or "",
            )
        )
    return out


def _search_weaviate(model_key: str, vector: list[float], top_k: int) -> list[dict[str, Any]]:
    import weaviate
    from weaviate.classes.query import MetadataQuery

    collection_name = f"Docs_{model_key}"
    client = weaviate.connect_to_local()
    try:
        col = client.collections.get(collection_name)
        results = col.query.near_vector(
            near_vector=vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        out = []
        for idx, obj in enumerate(results.objects, 1):
            props = obj.properties or {}
            distance = obj.metadata.distance if obj.metadata else None
            score = 1 - distance if distance is not None else 0.0
            out.append(_source_url_result(idx, props.get("source") or "", score, props.get("text") or ""))
        return out
    finally:
        client.close()
