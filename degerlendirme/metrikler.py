import math  # noqa: F401  (log2 için)


def recall_at_k(siralama, dogrular, k=10):
    if not dogrular:
        raise ValueError("dogrular bos olamaz")
    else:
        return len(set(siralama[:k]) & dogrular) / len(dogrular)
    
    raise NotImplementedError


def ndcg_at_k(siralama, dogrular, k=10):

    if not dogrular:
        raise ValueError("dogrular bos olamaz")
    else:
        dcg = 0
        for i, doc_id in enumerate(siralama[:k]):
            if doc_id in dogrular:
                dcg += 1 / math.log2(i + 2)

        ideal_dcg = 0
        for i in range(min(len(dogrular), k)):
            ideal_dcg += 1 / math.log2(i + 2)

        return dcg / ideal_dcg if ideal_dcg > 0 else 0
    raise NotImplementedError
