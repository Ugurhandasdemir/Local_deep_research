# Vektor DB Multi-Model Benchmark

## Ozet

_BENCHMARK OZETI_  
| Tarih: | 2026-04-29 20:48:34 |
|---|---|
| Mod: | full |
| Belge sayisi: | 52331 |
| Sorgu sayisi: | 10 |
| MODELLER: |  |
| e5_small | /home/ugo/Documents/Python/bitirememe projesi/models/e5_small/model |
| mpnet_multi | /home/ugo/Documents/Python/bitirememe projesi/models/mpnet_multi/model |
| e5_base | /home/ugo/Documents/Python/bitirememe projesi/models/e5_base/model |
| bge_squad | /home/ugo/Documents/Python/bitirememe projesi/models/bge_squad_model |
| qwen_lora | /home/ugo/Documents/Python/bitirememe projesi/models/qwen_lora |
| snowflake_arctic_l | /home/ugo/Documents/Python/bitirememe projesi/models/snowflake-arctic-embed-l-v2.0 |
| all_mini_l6 | /home/ugo/Documents/Python/bitirememe projesi/models/all_mini_l6_v2 |
| bge-m3-fine | /home/ugo/Documents/Python/bitirememe projesi/models/bge-m3-fine |
| e5_small_base | intfloat/multilingual-e5-small |
| mpnet_multi_base | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |
| e5_base_base | intfloat/multilingual-e5-base |
| bge_squad_base | BAAI/bge-large-en-v1.5 |
| qwen_lora_base | Qwen/Qwen3-Embedding-0.6B |
| snowflake_arctic_l_base | Snowflake/snowflake-arctic-embed-l-v2.0 |
| all_mini_l6_base | sentence-transformers/all-MiniLM-L6-v2 |
| bge_m3_base | BAAI/bge-m3 |
| minilm_l12 | sentence-transformers/all-MiniLM-L12-v2 |
| mpnet_base | sentence-transformers/all-mpnet-base-v2 |
| distilroberta | sentence-transformers/all-distilroberta-v1 |
| multi_qa_minilm | sentence-transformers/multi-qa-MiniLM-L6-cos-v1 |
| multi_qa_mpnet | sentence-transformers/multi-qa-mpnet-base-dot-v1 |
| paraphrase_multi | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| bge_small_en | BAAI/bge-small-en-v1.5 |
| bge_base_en | BAAI/bge-base-en-v1.5 |
| gte_small | thenlper/gte-small |
| gte_base | thenlper/gte-base |
| e5_small_hf | intfloat/e5-small-v2 |
| e5_base_hf | intfloat/e5-base-v2 |

## Yazma Sureleri

_MODEL x VERITABANI YAZMA SURELERI (saniye)_  
| Model | milvus | qdrant | chromadb | lancedb | weaviate |
|---|---|---|---|---|---|
| e5_small | 9.497 | 17.86 | 83.043 | 37.926 | 18.026 |
| mpnet_multi | 16.675 | 38.175 | 91.564 | 39.845 | 13.892 |
| e5_base | 17.019 | 38.778 | 93.833 | 39.16 | 16.709 |
| bge_squad | 10.25 | 41.31 | 97.134 | 39.963 | 16.629 |
| qwen_lora | - | - | - | - | - |
| snowflake_arctic_l | 16.743 | 46.602 | 96.533 | 40.692 | 17.505 |
| all_mini_l6 | 16.849 | 21.181 | 86.524 | 40.576 | 12.628 |
| bge-m3-fine | - | - | - | - | - |
| e5_small_base | 15.876 | 20.894 | 87.046 | 39.337 | 11.342 |
| mpnet_multi_base | 17.05 | 38.275 | 91.665 | 39.511 | 14.886 |
| e5_base_base | 16.945 | 39.095 | 90.923 | 39.748 | 16.17 |
| bge_squad_base | 10.398 | 46.972 | 94.387 | 40.078 | 15.385 |
| qwen_lora_base | - | - | - | - | - |
| snowflake_arctic_l_base | 10.306 | 40.205 | 87.356 | 39.948 | 16.011 |
| all_mini_l6_base | 16.755 | 22.227 | 88.04 | 38.992 | 11.006 |
| bge_m3_base | - | - | - | - | - |
| minilm_l12 | 16.358 | 21.859 | 87.768 | 38.984 | 10.81 |
| mpnet_base | 16.861 | 38.417 | 91.377 | 39.363 | 13.547 |
| distilroberta | 15.603 | 35.734 | 91.014 | 39.251 | 14.454 |
| multi_qa_minilm | 16.345 | 22.111 | 88.165 | 39.069 | 11.189 |
| multi_qa_mpnet | 17.02 | 37.929 | 90.128 | 39.006 | 13.846 |
| paraphrase_multi | 16.835 | 21.919 | 89.494 | 39.608 | 11.048 |
| bge_small_en | 17.492 | 23.243 | 87.318 | 39.415 | 12.339 |
| bge_base_en | 16.441 | 38.192 | 89.149 | 39.196 | 13.81 |
| gte_small | 16.136 | 21.242 | 86.052 | 38.784 | 11.24 |
| gte_base | 17.183 | 36.246 | 88.571 | 38.671 | 15.753 |
| e5_small_hf | 16.64 | 22.047 | 86.177 | 39.125 | 11.762 |
| e5_base_hf | 16.866 | 37.911 | 91.178 | 38.841 | 14.872 |

## Arama Sonuclari

_TUM ARAMA TESTLERI (ms)_  
| Sira | Veritabani | Model | Algoritma | Ort | Min | Max | Std | P50 | P95 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | chromadb | minilm_l12 | HNSW_batch | 2.955 | 2.868 | 3.083 | 0.058 | 2.954 | 3.044 |
| 2 | chromadb | all_mini_l6_base | HNSW_batch | 3.187 | 3.103 | 3.343 | 0.077 | 3.154 | 3.311 |
| 3 | chromadb | gte_small | HNSW_batch | 3.268 | 3.177 | 3.52 | 0.11 | 3.227 | 3.485 |
| 4 | chromadb | bge_small_en | HNSW_batch | 3.492 | 3.389 | 3.639 | 0.075 | 3.488 | 3.605 |
| 5 | chromadb | multi_qa_minilm | HNSW_batch | 3.501 | 3.407 | 3.65 | 0.077 | 3.473 | 3.633 |
| 6 | chromadb | paraphrase_multi | HNSW_batch | 3.527 | 3.446 | 3.635 | 0.061 | 3.518 | 3.62 |
| 7 | chromadb | e5_small_base | HNSW_batch | 3.571 | 3.463 | 3.694 | 0.068 | 3.568 | 3.676 |
| 8 | chromadb | e5_small_hf | HNSW_batch | 3.773 | 3.689 | 3.885 | 0.071 | 3.755 | 3.874 |
| 9 | chromadb | all_mini_l6 | HNSW_batch | 3.808 | 3.669 | 3.954 | 0.095 | 3.774 | 3.952 |
| 10 | chromadb | e5_small | HNSW_batch | 4.355 | 4.223 | 4.449 | 0.082 | 4.366 | 4.447 |
| 11 | chromadb | mpnet_base | HNSW_batch | 4.528 | 4.304 | 4.913 | 0.163 | 4.504 | 4.792 |
| 12 | chromadb | distilroberta | HNSW_batch | 4.665 | 4.536 | 4.87 | 0.116 | 4.648 | 4.845 |
| 13 | chromadb | bge_base_en | HNSW_batch | 5.206 | 4.992 | 5.945 | 0.284 | 5.09 | 5.718 |
| 14 | chromadb | gte_base | HNSW_batch | 5.207 | 5.088 | 5.349 | 0.086 | 5.176 | 5.33 |
| 15 | chromadb | mpnet_multi_base | HNSW_batch | 5.337 | 5.262 | 5.483 | 0.069 | 5.31 | 5.453 |
| 16 | chromadb | multi_qa_mpnet | HNSW_batch | 5.711 | 5.582 | 5.846 | 0.083 | 5.695 | 5.843 |
| 17 | chromadb | e5_base_hf | HNSW_batch | 5.783 | 5.669 | 6.171 | 0.137 | 5.732 | 6.005 |
| 18 | chromadb | mpnet_multi | HNSW_batch | 5.822 | 5.665 | 5.979 | 0.107 | 5.813 | 5.971 |
| 19 | chromadb | e5_base_base | HNSW_batch | 6.182 | 5.931 | 6.49 | 0.17 | 6.222 | 6.409 |
| 20 | chromadb | minilm_l12 | HNSW_n5 | 6.397 | 6.149 | 6.667 | 0.152 | 6.405 | 6.614 |
| 21 | chromadb | paraphrase_multi | HNSW_n5 | 6.798 | 6.61 | 6.884 | 0.073 | 6.813 | 6.871 |
| 22 | chromadb | gte_small | HNSW_n5 | 6.821 | 6.692 | 6.948 | 0.095 | 6.794 | 6.944 |
| 23 | chromadb | snowflake_arctic_l_base | HNSW_batch | 6.829 | 6.748 | 6.911 | 0.055 | 6.831 | 6.906 |
| 24 | chromadb | all_mini_l6_base | HNSW_n5 | 6.962 | 6.781 | 7.431 | 0.178 | 6.909 | 7.268 |
| 25 | chromadb | e5_base | HNSW_batch | 7.029 | 6.871 | 7.252 | 0.113 | 7.024 | 7.209 |
| 26 | chromadb | bge_squad_base | HNSW_batch | 7.104 | 6.854 | 8.012 | 0.316 | 7.02 | 7.613 |
| 27 | chromadb | e5_small_base | HNSW_n5 | 7.121 | 6.886 | 7.249 | 0.089 | 7.128 | 7.215 |
| 28 | chromadb | multi_qa_minilm | HNSW_n5 | 7.371 | 7.048 | 7.652 | 0.173 | 7.373 | 7.608 |
| 29 | chromadb | bge_squad | HNSW_batch | 7.511 | 6.836 | 9.429 | 0.806 | 7.126 | 9.093 |
| 30 | chromadb | multi_qa_minilm | HNSW_default | 7.571 | 7.035 | 8.598 | 0.474 | 7.507 | 8.361 |
| 31 | chromadb | bge_small_en | HNSW_n5 | 7.574 | 6.827 | 8.094 | 0.426 | 7.697 | 8.065 |
| 32 | chromadb | paraphrase_multi | HNSW_default | 7.614 | 7 | 8.181 | 0.494 | 7.661 | 8.174 |
| 33 | chromadb | gte_small | HNSW_default | 7.629 | 7.267 | 8.034 | 0.259 | 7.574 | 8.017 |
| 34 | chromadb | e5_small_hf | HNSW_n5 | 7.633 | 7.206 | 8.739 | 0.442 | 7.443 | 8.394 |
| 35 | chromadb | all_mini_l6_base | HNSW_default | 7.65 | 7.5 | 7.898 | 0.121 | 7.63 | 7.872 |
| 36 | chromadb | e5_small_base | HNSW_default | 7.766 | 7.605 | 8.026 | 0.145 | 7.715 | 8.026 |
| 37 | chromadb | e5_small | HNSW_n5 | 7.831 | 7.429 | 8.598 | 0.344 | 7.766 | 8.405 |
| 38 | chromadb | all_mini_l6 | HNSW_n5 | 7.887 | 7.642 | 8.125 | 0.161 | 7.85 | 8.099 |
| 39 | chromadb | minilm_l12 | HNSW_default | 7.944 | 7.129 | 10.155 | 0.906 | 7.721 | 9.618 |
| 40 | chromadb | bge_small_en | HNSW_default | 7.973 | 7.375 | 8.8 | 0.463 | 7.808 | 8.747 |
| 41 | weaviate | minilm_l12 | HNSW_limit5 | 7.979 | 7.593 | 9.641 | 0.577 | 7.787 | 8.958 |
| 42 | chromadb | snowflake_arctic_l | HNSW_batch | 8.038 | 7.913 | 8.146 | 0.069 | 8.048 | 8.124 |
| 43 | weaviate | all_mini_l6_base | HNSW_limit5 | 8.062 | 7.865 | 8.593 | 0.227 | 7.966 | 8.488 |
| 44 | chromadb | minilm_l12 | HNSW_n20 | 8.143 | 7.837 | 8.879 | 0.267 | 8.074 | 8.607 |
| 45 | weaviate | paraphrase_multi | HNSW_limit5 | 8.143 | 7.804 | 8.833 | 0.291 | 8.097 | 8.654 |
| 46 | chromadb | e5_small_hf | HNSW_default | 8.162 | 7.722 | 9.217 | 0.455 | 7.979 | 9.008 |
| 47 | chromadb | mpnet_base | HNSW_n5 | 8.214 | 7.87 | 9.413 | 0.42 | 8.09 | 8.924 |
| 48 | chromadb | distilroberta | HNSW_n5 | 8.24 | 7.916 | 9.004 | 0.287 | 8.156 | 8.728 |
| 49 | chromadb | e5_small | HNSW_default | 8.503 | 8.005 | 9.582 | 0.482 | 8.227 | 9.295 |
| 50 | chromadb | all_mini_l6 | HNSW_default | 8.597 | 8.175 | 10.643 | 0.714 | 8.323 | 9.879 |
| 51 | weaviate | bge_small_en | HNSW_limit5 | 8.61 | 8.3 | 9.07 | 0.269 | 8.484 | 9.035 |
| 52 | chromadb | bge_base_en | HNSW_n5 | 8.614 | 8.486 | 8.739 | 0.08 | 8.625 | 8.736 |
| 53 | chromadb | gte_base | HNSW_n5 | 8.658 | 8.313 | 9.1 | 0.236 | 8.607 | 9.022 |
| 54 | chromadb | gte_small | HNSW_n20 | 8.697 | 8.342 | 9.032 | 0.228 | 8.749 | 9.001 |
| 55 | weaviate | gte_small | HNSW_limit5 | 8.708 | 8.363 | 9.369 | 0.314 | 8.592 | 9.296 |
| 56 | weaviate | mpnet_base | HNSW_limit5 | 8.87 | 8.58 | 9.192 | 0.207 | 8.871 | 9.171 |
| 57 | weaviate | minilm_l12 | HNSW_default | 8.872 | 8.541 | 10.021 | 0.414 | 8.757 | 9.607 |
| 58 | chromadb | bge_small_en | HNSW_n20 | 8.9 | 8.727 | 9.027 | 0.095 | 8.899 | 9.015 |
| 59 | chromadb | distilroberta | HNSW_default | 8.947 | 8.613 | 9.628 | 0.275 | 8.885 | 9.426 |
| 60 | chromadb | mpnet_multi_base | HNSW_n5 | 8.975 | 8.782 | 9.171 | 0.113 | 8.96 | 9.137 |
| 61 | chromadb | multi_qa_minilm | HNSW_n20 | 9.042 | 8.767 | 9.405 | 0.173 | 9.037 | 9.319 |
| 62 | chromadb | mpnet_base | HNSW_default | 9.078 | 8.758 | 9.324 | 0.215 | 9.146 | 9.319 |
| 63 | chromadb | bge_base_en | HNSW_default | 9.1 | 8.818 | 9.274 | 0.141 | 9.154 | 9.252 |
| 64 | chromadb | all_mini_l6_base | HNSW_n20 | 9.136 | 8.626 | 11.357 | 0.756 | 8.908 | 10.411 |
| 65 | chromadb | e5_base_base | HNSW_n5 | 9.136 | 8.894 | 9.422 | 0.153 | 9.161 | 9.362 |
| 66 | weaviate | all_mini_l6_base | HNSW_default | 9.147 | 8.573 | 11.003 | 0.749 | 8.839 | 10.603 |
| 67 | chromadb | all_mini_l6 | HNSW_n20 | 9.237 | 8.885 | 9.681 | 0.26 | 9.293 | 9.595 |
| 68 | chromadb | mpnet_multi | HNSW_n5 | 9.272 | 9.123 | 9.593 | 0.138 | 9.273 | 9.491 |
| 69 | chromadb | gte_base | HNSW_default | 9.326 | 9.042 | 10.01 | 0.268 | 9.265 | 9.788 |
| 70 | chromadb | paraphrase_multi | HNSW_n20 | 9.371 | 9.038 | 10.742 | 0.471 | 9.23 | 10.164 |
| 71 | weaviate | e5_small_base | HNSW_limit5 | 9.424 | 9.114 | 10.246 | 0.326 | 9.317 | 9.997 |
| 72 | chromadb | e5_base_hf | HNSW_n5 | 9.43 | 9.166 | 9.772 | 0.192 | 9.403 | 9.746 |
| 73 | chromadb | multi_qa_mpnet | HNSW_n5 | 9.45 | 9.277 | 9.639 | 0.122 | 9.462 | 9.616 |
| 74 | weaviate | distilroberta | HNSW_limit5 | 9.45 | 9.22 | 10.195 | 0.284 | 9.302 | 9.944 |
| 75 | chromadb | e5_small_base | HNSW_n20 | 9.453 | 8.889 | 10.873 | 0.541 | 9.257 | 10.369 |
| 76 | weaviate | e5_small_hf | HNSW_limit5 | 9.461 | 9.216 | 9.722 | 0.174 | 9.461 | 9.711 |
| 77 | chromadb | e5_small | HNSW_n20 | 9.587 | 9.084 | 10.296 | 0.448 | 9.363 | 10.208 |
| 78 | chromadb | mpnet_multi_base | HNSW_default | 9.678 | 9.446 | 10.139 | 0.19 | 9.631 | 9.997 |
| 79 | weaviate | bge_base_en | HNSW_limit5 | 9.697 | 9.23 | 9.987 | 0.221 | 9.753 | 9.961 |
| 80 | milvus | e5_small | HNSW_batch | 9.734 | 9.152 | 10.187 | 0.285 | 9.773 | 10.097 |
| 81 | weaviate | paraphrase_multi | HNSW_default | 9.761 | 9.237 | 10.371 | 0.433 | 9.651 | 10.368 |
| 82 | weaviate | bge_small_en | BM25 | 9.779 | 9.537 | 10.084 | 0.159 | 9.764 | 10.022 |
| 83 | chromadb | e5_base_base | HNSW_default | 9.784 | 9.524 | 10.001 | 0.15 | 9.786 | 9.983 |
| 84 | weaviate | multi_qa_minilm | HNSW_limit5 | 9.793 | 9.519 | 10.478 | 0.277 | 9.709 | 10.307 |
| 85 | chromadb | e5_small_hf | HNSW_n20 | 9.823 | 9.51 | 10.118 | 0.213 | 9.855 | 10.113 |
| 86 | milvus | paraphrase_multi | HNSW_batch | 9.824 | 9.362 | 10.278 | 0.281 | 9.787 | 10.216 |
| 87 | weaviate | gte_small | HNSW_default | 9.912 | 9.449 | 11.277 | 0.578 | 9.708 | 11.045 |
| 88 | weaviate | mpnet_base | HNSW_default | 9.927 | 9.175 | 11.77 | 0.681 | 9.723 | 11.046 |
| 89 | weaviate | e5_small_hf | BM25 | 9.929 | 9.614 | 10.719 | 0.306 | 9.865 | 10.445 |
| 90 | weaviate | gte_base | BM25 | 9.933 | 9.716 | 10.408 | 0.187 | 9.888 | 10.279 |
| 91 | milvus | all_mini_l6_base | HNSW_batch | 9.945 | 9.59 | 10.577 | 0.303 | 9.858 | 10.451 |
| 92 | weaviate | gte_small | BM25 | 9.958 | 9.813 | 10.362 | 0.162 | 9.898 | 10.264 |
| 93 | weaviate | distilroberta | BM25 | 9.964 | 9.796 | 10.338 | 0.209 | 9.858 | 10.309 |
| 94 | milvus | e5_small_hf | HNSW_batch | 9.973 | 9.61 | 10.324 | 0.254 | 9.973 | 10.321 |
| 95 | weaviate | e5_base_hf | BM25 | 9.984 | 9.684 | 10.298 | 0.172 | 9.948 | 10.289 |
| 96 | weaviate | multi_qa_minilm | BM25 | 9.986 | 9.792 | 10.606 | 0.226 | 9.936 | 10.355 |
| 97 | chromadb | bge_squad_base | HNSW_n5 | 9.987 | 9.657 | 10.311 | 0.188 | 9.996 | 10.237 |
| 98 | weaviate | multi_qa_mpnet | BM25 | 9.993 | 9.7 | 10.463 | 0.219 | 9.974 | 10.368 |
| 99 | weaviate | bge_base_en | BM25 | 9.994 | 9.791 | 10.249 | 0.164 | 9.916 | 10.247 |
| 100 | weaviate | e5_small | BM25 | 10.012 | 9.529 | 11.074 | 0.422 | 9.906 | 10.764 |
| 101 | weaviate | snowflake_arctic_l_base | BM25 | 10.016 | 9.709 | 10.814 | 0.285 | 9.965 | 10.47 |
| 102 | milvus | e5_small_base | HNSW_batch | 10.019 | 9.713 | 10.648 | 0.293 | 9.881 | 10.53 |
| 103 | chromadb | mpnet_multi | HNSW_default | 10.036 | 9.786 | 10.721 | 0.247 | 9.96 | 10.439 |
| 104 | weaviate | paraphrase_multi | BM25 | 10.038 | 9.854 | 10.425 | 0.193 | 9.941 | 10.389 |
| 105 | milvus | all_mini_l6 | HNSW_batch | 10.043 | 9.546 | 10.702 | 0.302 | 10.022 | 10.506 |
| 106 | weaviate | bge_small_en | HNSW_default | 10.06 | 9.666 | 11.02 | 0.38 | 9.982 | 10.687 |
| 107 | weaviate | bge_squad_base | BM25 | 10.105 | 9.931 | 10.618 | 0.192 | 10.102 | 10.413 |
| 108 | weaviate | distilroberta | HNSW_default | 10.109 | 9.535 | 12.249 | 0.76 | 9.781 | 11.395 |
| 109 | milvus | gte_small | HNSW_batch | 10.15 | 9.329 | 11.49 | 0.575 | 10.078 | 11.091 |
| 110 | weaviate | minilm_l12 | BM25 | 10.15 | 9.971 | 10.684 | 0.217 | 10.058 | 10.525 |
| 111 | chromadb | e5_base | HNSW_n5 | 10.154 | 9.906 | 10.366 | 0.175 | 10.216 | 10.353 |
| 112 | chromadb | e5_base_hf | HNSW_default | 10.171 | 9.878 | 10.589 | 0.243 | 10.143 | 10.543 |
| 113 | weaviate | gte_base | HNSW_limit5 | 10.188 | 9.845 | 10.825 | 0.274 | 10.257 | 10.59 |
| 114 | weaviate | e5_base_base | BM25 | 10.221 | 9.873 | 11.235 | 0.447 | 10.027 | 11.113 |
| 115 | weaviate | e5_small_base | BM25 | 10.271 | 9.953 | 10.769 | 0.272 | 10.196 | 10.692 |
| 116 | weaviate | bge_squad | BM25 | 10.287 | 10.044 | 10.591 | 0.182 | 10.256 | 10.564 |
| 117 | milvus | bge_small_en | HNSW_batch | 10.292 | 10.063 | 10.483 | 0.149 | 10.286 | 10.478 |
| 118 | milvus | multi_qa_minilm | HNSW_batch | 10.302 | 9.747 | 10.695 | 0.297 | 10.38 | 10.68 |
| 119 | chromadb | multi_qa_mpnet | HNSW_default | 10.304 | 9.872 | 10.674 | 0.196 | 10.316 | 10.577 |
| 120 | weaviate | all_mini_l6_base | BM25 | 10.329 | 10.098 | 10.879 | 0.222 | 10.277 | 10.722 |
| 121 | weaviate | mpnet_base | BM25 | 10.329 | 9.963 | 11.651 | 0.462 | 10.2 | 11.099 |
| 122 | chromadb | distilroberta | HNSW_n20 | 10.332 | 9.959 | 10.898 | 0.267 | 10.279 | 10.775 |
| 123 | milvus | minilm_l12 | HNSW_batch | 10.346 | 9.99 | 10.578 | 0.176 | 10.409 | 10.537 |
| 124 | chromadb | mpnet_base | HNSW_n20 | 10.347 | 10.184 | 10.502 | 0.102 | 10.362 | 10.483 |
| 125 | weaviate | mpnet_multi_base | HNSW_limit5 | 10.4 | 9.917 | 11.843 | 0.543 | 10.182 | 11.322 |
| 126 | weaviate | snowflake_arctic_l | BM25 | 10.435 | 10.12 | 11.008 | 0.265 | 10.355 | 10.906 |
| 127 | chromadb | gte_base | HNSW_n20 | 10.573 | 10.236 | 10.971 | 0.23 | 10.552 | 10.952 |
| 128 | weaviate | e5_base | BM25 | 10.575 | 10.315 | 11.2 | 0.268 | 10.512 | 11.07 |
| 129 | chromadb | bge_base_en | HNSW_n20 | 10.671 | 10.354 | 10.936 | 0.17 | 10.667 | 10.917 |
| 130 | weaviate | mpnet_multi_base | BM25 | 10.694 | 10.359 | 11.735 | 0.405 | 10.492 | 11.408 |
| 131 | weaviate | multi_qa_mpnet | HNSW_limit5 | 10.782 | 10.464 | 11.333 | 0.23 | 10.755 | 11.163 |
| 132 | chromadb | snowflake_arctic_l_base | HNSW_n5 | 10.802 | 10.324 | 11.277 | 0.268 | 10.803 | 11.175 |
| 133 | weaviate | all_mini_l6 | HNSW_limit5 | 10.828 | 10.143 | 12.049 | 0.503 | 10.8 | 11.626 |
| 134 | weaviate | e5_small_hf | HNSW_default | 10.83 | 10.361 | 11.771 | 0.426 | 10.708 | 11.637 |
| 135 | weaviate | all_mini_l6 | BM25 | 10.839 | 10.407 | 11.846 | 0.442 | 10.697 | 11.676 |
| 136 | chromadb | bge_squad | HNSW_default | 10.906 | 10.423 | 11.409 | 0.318 | 11.021 | 11.273 |
| 137 | weaviate | mpnet_multi | BM25 | 11.014 | 10.328 | 12.665 | 0.782 | 10.533 | 12.358 |
| 138 | chromadb | bge_squad | HNSW_n5 | 11.068 | 10.458 | 13.387 | 0.876 | 10.695 | 12.688 |
| 139 | weaviate | e5_small_base | HNSW_default | 11.073 | 10.68 | 11.931 | 0.386 | 10.935 | 11.724 |
| 140 | chromadb | mpnet_multi_base | HNSW_n20 | 11.082 | 10.839 | 11.644 | 0.231 | 11.022 | 11.47 |
| 141 | weaviate | bge_base_en | HNSW_default | 11.092 | 10.503 | 12.136 | 0.5 | 10.876 | 12.026 |
| 142 | chromadb | bge_squad_base | HNSW_default | 11.108 | 10.54 | 11.917 | 0.441 | 10.934 | 11.834 |
| 143 | chromadb | snowflake_arctic_l_base | HNSW_default | 11.191 | 10.627 | 11.611 | 0.331 | 11.333 | 11.572 |
| 144 | chromadb | e5_base_base | HNSW_n20 | 11.202 | 10.862 | 11.597 | 0.244 | 11.175 | 11.558 |
| 145 | chromadb | e5_base_hf | HNSW_n20 | 11.251 | 10.836 | 11.606 | 0.243 | 11.238 | 11.573 |
| 146 | chromadb | snowflake_arctic_l | HNSW_n5 | 11.264 | 11.017 | 11.534 | 0.228 | 11.263 | 11.527 |
| 147 | chromadb | mpnet_multi | HNSW_n20 | 11.32 | 11.13 | 11.6 | 0.183 | 11.229 | 11.58 |
| 148 | weaviate | snowflake_arctic_l_base | HNSW_limit5 | 11.336 | 10.847 | 11.965 | 0.29 | 11.349 | 11.783 |
| 149 | weaviate | e5_base_hf | HNSW_limit5 | 11.362 | 11.133 | 11.714 | 0.166 | 11.32 | 11.64 |
| 150 | weaviate | e5_base_base | HNSW_limit5 | 11.386 | 10.825 | 13.033 | 0.578 | 11.218 | 12.346 |
| 151 | chromadb | e5_base | HNSW_default | 11.419 | 10.43 | 13.407 | 0.946 | 10.935 | 13.109 |
| 152 | weaviate | gte_base | HNSW_default | 11.447 | 11.079 | 12.24 | 0.384 | 11.317 | 12.058 |
| 153 | chromadb | minilm_l12 | HNSW_n50 | 11.552 | 11.055 | 12.769 | 0.628 | 11.24 | 12.753 |
| 154 | weaviate | e5_small | HNSW_limit5 | 11.628 | 11.18 | 12.209 | 0.294 | 11.694 | 12.055 |
| 155 | chromadb | all_mini_l6_base | HNSW_n50 | 11.713 | 11.557 | 11.93 | 0.101 | 11.716 | 11.865 |
| 156 | chromadb | gte_small | HNSW_n50 | 11.762 | 11.459 | 12.069 | 0.203 | 11.76 | 12.026 |
| 157 | weaviate | bge_squad_base | HNSW_limit5 | 11.901 | 11.349 | 13.315 | 0.62 | 11.606 | 13.053 |
| 158 | weaviate | all_mini_l6 | HNSW_default | 11.93 | 11.137 | 13.052 | 0.592 | 11.882 | 12.851 |
| 159 | chromadb | multi_qa_mpnet | HNSW_n20 | 11.934 | 11.469 | 12.326 | 0.236 | 11.955 | 12.278 |
| 160 | chromadb | all_mini_l6 | HNSW_n50 | 11.935 | 11.659 | 12.583 | 0.244 | 11.858 | 12.35 |
| 161 | chromadb | bge_squad | HNSW_n20 | 11.963 | 11.656 | 12.166 | 0.153 | 11.974 | 12.15 |
| 162 | chromadb | bge_small_en | HNSW_n50 | 12.014 | 11.755 | 12.562 | 0.25 | 11.901 | 12.485 |
| 163 | chromadb | e5_small_base | HNSW_n50 | 12.146 | 11.694 | 12.939 | 0.423 | 11.923 | 12.846 |
| 164 | weaviate | bge_squad | HNSW_limit5 | 12.155 | 11.775 | 13.131 | 0.428 | 11.994 | 12.999 |
| 165 | weaviate | multi_qa_mpnet | HNSW_default | 12.164 | 11.639 | 13.402 | 0.64 | 11.851 | 13.357 |
| 166 | chromadb | bge_squad_base | HNSW_n20 | 12.189 | 11.758 | 12.448 | 0.215 | 12.256 | 12.431 |
| 167 | chromadb | snowflake_arctic_l | HNSW_default | 12.222 | 11.998 | 12.514 | 0.13 | 12.239 | 12.414 |
| 168 | weaviate | mpnet_multi | HNSW_limit5 | 12.324 | 11.677 | 13.198 | 0.511 | 12.28 | 13.09 |
| 169 | weaviate | mpnet_multi_base | HNSW_default | 12.414 | 11.925 | 13.336 | 0.39 | 12.288 | 13.121 |
| 170 | chromadb | e5_base | HNSW_n20 | 12.423 | 12.194 | 12.7 | 0.158 | 12.472 | 12.626 |
| 171 | weaviate | e5_base_hf | HNSW_default | 12.461 | 12.183 | 13.206 | 0.29 | 12.376 | 12.963 |
| 172 | weaviate | e5_base_base | HNSW_default | 12.488 | 12.114 | 13.771 | 0.481 | 12.293 | 13.387 |
| 173 | weaviate | all_mini_l6_base | HNSW_limit20 | 12.795 | 12.141 | 14.048 | 0.501 | 12.686 | 13.683 |
| 174 | weaviate | snowflake_arctic_l_base | HNSW_default | 12.839 | 12.354 | 13.787 | 0.419 | 12.663 | 13.603 |
| 175 | weaviate | mpnet_multi | HNSW_default | 12.993 | 12.443 | 14.692 | 0.651 | 12.747 | 14.244 |
| 176 | chromadb | e5_small_hf | HNSW_n50 | 13.005 | 12.721 | 13.532 | 0.224 | 12.941 | 13.431 |
| 177 | weaviate | bge_squad | HNSW_default | 13.039 | 12.743 | 13.575 | 0.246 | 12.956 | 13.455 |
| 178 | weaviate | bge_squad_base | HNSW_default | 13.059 | 12.198 | 13.764 | 0.571 | 13.149 | 13.735 |
| 179 | chromadb | snowflake_arctic_l | HNSW_n20 | 13.1 | 12.663 | 14.05 | 0.404 | 12.991 | 13.873 |
| 180 | weaviate | snowflake_arctic_l | HNSW_limit5 | 13.166 | 12.669 | 13.831 | 0.409 | 13.062 | 13.772 |
| 181 | chromadb | multi_qa_minilm | HNSW_n50 | 13.191 | 12.747 | 13.779 | 0.328 | 13.176 | 13.766 |
| 182 | chromadb | paraphrase_multi | HNSW_n50 | 13.256 | 12.847 | 15.108 | 0.631 | 13.1 | 14.285 |
| 183 | chromadb | e5_small | HNSW_n50 | 13.433 | 13.189 | 13.806 | 0.176 | 13.372 | 13.733 |
| 184 | chromadb | snowflake_arctic_l_base | HNSW_n20 | 13.511 | 12.542 | 15.14 | 0.713 | 13.319 | 14.746 |
| 185 | weaviate | minilm_l12 | HNSW_limit20 | 13.599 | 12.85 | 14.874 | 0.632 | 13.405 | 14.57 |
| 186 | weaviate | bge_small_en | HNSW_limit20 | 13.722 | 13.31 | 14.252 | 0.334 | 13.619 | 14.194 |
| 187 | weaviate | paraphrase_multi | HNSW_limit20 | 13.745 | 13.168 | 15.648 | 0.69 | 13.466 | 14.953 |
| 188 | weaviate | e5_small | HNSW_default | 14.046 | 12.439 | 15.712 | 0.801 | 14.019 | 15.219 |
| 189 | weaviate | distilroberta | HNSW_limit20 | 14.179 | 13.601 | 15.514 | 0.667 | 13.811 | 15.3 |
| 190 | chromadb | distilroberta | HNSW_n50 | 14.196 | 13.453 | 16.522 | 0.839 | 13.871 | 15.694 |
| 191 | chromadb | mpnet_base | HNSW_n50 | 14.198 | 13.882 | 14.46 | 0.182 | 14.217 | 14.428 |
| 192 | chromadb | bge_base_en | HNSW_n50 | 14.362 | 13.938 | 15.971 | 0.555 | 14.262 | 15.255 |
| 193 | weaviate | e5_base | HNSW_limit5 | 14.402 | 13.834 | 15.821 | 0.582 | 14.257 | 15.52 |
| 194 | chromadb | gte_base | HNSW_n50 | 14.42 | 13.292 | 16.226 | 0.801 | 14.375 | 15.752 |
| 195 | weaviate | mpnet_base | HNSW_limit20 | 14.512 | 13.814 | 14.992 | 0.319 | 14.565 | 14.939 |
| 196 | weaviate | multi_qa_minilm | HNSW_default | 14.763 | 10.815 | 26.618 | 5.479 | 12.455 | 25.655 |
| 197 | chromadb | mpnet_multi_base | HNSW_n50 | 14.797 | 14.557 | 15.091 | 0.162 | 14.834 | 15.027 |
| 198 | weaviate | snowflake_arctic_l | HNSW_default | 14.939 | 14.167 | 16.514 | 0.672 | 14.805 | 16.143 |
| 199 | chromadb | bge_squad | HNSW_n50 | 15.3 | 15.068 | 15.532 | 0.116 | 15.316 | 15.472 |
| 200 | chromadb | mpnet_multi | HNSW_n50 | 15.307 | 15.024 | 15.863 | 0.23 | 15.209 | 15.691 |
| 201 | chromadb | e5_base_hf | HNSW_n50 | 15.35 | 15.141 | 15.547 | 0.111 | 15.347 | 15.518 |
| 202 | chromadb | multi_qa_mpnet | HNSW_n50 | 15.487 | 15.071 | 15.852 | 0.211 | 15.502 | 15.78 |
| 203 | chromadb | e5_base_base | HNSW_n50 | 15.551 | 15.204 | 16.333 | 0.345 | 15.445 | 16.136 |
| 204 | weaviate | e5_base | HNSW_default | 15.551 | 14.938 | 16.473 | 0.505 | 15.428 | 16.421 |
| 205 | weaviate | gte_small | HNSW_limit20 | 15.672 | 14.531 | 16.5 | 0.722 | 16.02 | 16.479 |
| 206 | weaviate | multi_qa_minilm | HNSW_limit20 | 15.92 | 14.422 | 16.625 | 0.687 | 16.112 | 16.624 |
| 207 | chromadb | snowflake_arctic_l_base | HNSW_n50 | 16.081 | 15.893 | 16.306 | 0.131 | 16.042 | 16.287 |
| 208 | weaviate | bge_base_en | HNSW_limit20 | 16.151 | 15.879 | 16.69 | 0.28 | 16.028 | 16.688 |
| 209 | chromadb | e5_base | HNSW_n50 | 16.401 | 15.73 | 17.218 | 0.401 | 16.286 | 17.118 |
| 210 | chromadb | bge_squad_base | HNSW_n50 | 16.41 | 15.544 | 17.753 | 0.612 | 16.419 | 17.409 |
| 211 | milvus | e5_base_base | HNSW_batch | 16.451 | 16.151 | 17.253 | 0.295 | 16.377 | 16.931 |
| 212 | weaviate | mpnet_multi_base | HNSW_limit20 | 16.54 | 15.429 | 19.411 | 1.101 | 16.207 | 18.44 |
| 213 | weaviate | e5_small_hf | HNSW_limit20 | 16.556 | 15.946 | 16.947 | 0.318 | 16.578 | 16.938 |
| 214 | milvus | multi_qa_mpnet | HNSW_batch | 16.609 | 16.122 | 17.174 | 0.405 | 16.477 | 17.163 |
| 215 | weaviate | e5_small_base | HNSW_limit20 | 16.733 | 15.028 | 19.262 | 1.332 | 16.611 | 18.683 |
| 216 | milvus | bge_base_en | HNSW_batch | 16.8 | 16.259 | 17.846 | 0.473 | 16.624 | 17.602 |
| 217 | milvus | e5_base | HNSW_batch | 16.841 | 16.009 | 17.571 | 0.437 | 16.786 | 17.507 |
| 218 | weaviate | gte_base | HNSW_limit20 | 17.009 | 16.293 | 18.046 | 0.491 | 16.911 | 17.812 |
| 219 | milvus | mpnet_multi_base | HNSW_batch | 17.062 | 16.537 | 18.339 | 0.555 | 16.839 | 18.009 |
| 220 | chromadb | snowflake_arctic_l | HNSW_n50 | 17.253 | 16.235 | 19.153 | 0.916 | 16.814 | 18.892 |
| 221 | milvus | e5_base_hf | HNSW_batch | 17.386 | 16.903 | 18.08 | 0.416 | 17.176 | 17.983 |
| 222 | weaviate | multi_qa_mpnet | HNSW_limit20 | 17.548 | 16.688 | 18.217 | 0.444 | 17.634 | 18.084 |
| 223 | milvus | mpnet_base | HNSW_batch | 17.591 | 16.737 | 18.472 | 0.64 | 17.722 | 18.376 |
| 224 | milvus | mpnet_multi | HNSW_batch | 17.696 | 16.935 | 19.322 | 0.669 | 17.425 | 18.864 |
| 225 | milvus | distilroberta | HNSW_batch | 17.811 | 16.094 | 26.755 | 3.009 | 17.024 | 22.52 |
| 226 | milvus | gte_base | HNSW_batch | 17.839 | 17.273 | 19.001 | 0.434 | 17.78 | 18.519 |
| 227 | weaviate | all_mini_l6 | HNSW_limit20 | 18.16 | 16.642 | 19.719 | 0.895 | 18.232 | 19.44 |
| 228 | weaviate | e5_base_base | HNSW_limit20 | 18.721 | 17.748 | 20.693 | 0.845 | 18.521 | 20.287 |
| 229 | weaviate | bge_squad_base | HNSW_limit20 | 18.944 | 17.942 | 20.756 | 0.726 | 18.921 | 20.11 |
| 230 | weaviate | e5_small | HNSW_limit20 | 18.982 | 17.914 | 21.221 | 1.052 | 18.674 | 20.742 |
| 231 | weaviate | snowflake_arctic_l_base | HNSW_limit20 | 19.012 | 18.034 | 20.411 | 0.99 | 18.44 | 20.333 |
| 232 | weaviate | e5_base_hf | HNSW_limit20 | 19.067 | 18.434 | 20.95 | 0.72 | 18.852 | 20.301 |
| 233 | weaviate | bge_squad | HNSW_limit20 | 19.48 | 18.707 | 20.658 | 0.544 | 19.513 | 20.307 |
| 234 | qdrant | mpnet_multi | HNSW_ef32 | 19.555 | 19.214 | 20.184 | 0.284 | 19.493 | 20.085 |
| 235 | weaviate | mpnet_multi | HNSW_limit20 | 20.364 | 19.054 | 22.137 | 1.177 | 19.744 | 22.049 |
| 236 | qdrant | gte_base | HNSW_ef32 | 20.389 | 19.821 | 20.651 | 0.25 | 20.477 | 20.633 |
| 237 | qdrant | mpnet_base | HNSW_ef32 | 20.528 | 20.096 | 21.184 | 0.31 | 20.529 | 21.032 |
| 238 | qdrant | e5_base_base | HNSW_ef32 | 20.864 | 20.405 | 21.98 | 0.459 | 20.68 | 21.685 |
| 239 | weaviate | snowflake_arctic_l | HNSW_limit20 | 21.352 | 20.132 | 23.084 | 0.759 | 21.333 | 22.572 |
| 240 | qdrant | bge_base_en | HNSW_ef32 | 21.606 | 21.29 | 21.855 | 0.187 | 21.676 | 21.818 |
| 241 | qdrant | distilroberta | HNSW_ef32 | 21.631 | 21.18 | 22.196 | 0.324 | 21.609 | 22.188 |
| 242 | qdrant | bge_squad_base | HNSW_ef32 | 21.73 | 21.369 | 22.4 | 0.294 | 21.703 | 22.251 |
| 243 | milvus | snowflake_arctic_l_base | HNSW_batch | 21.906 | 20.736 | 23.185 | 0.885 | 21.622 | 23.061 |
| 244 | milvus | bge_squad_base | HNSW_batch | 22.244 | 20.491 | 24.527 | 1.224 | 21.949 | 24.315 |
| 245 | qdrant | snowflake_arctic_l_base | HNSW_ef32 | 22.332 | 21.836 | 23.257 | 0.46 | 22.143 | 23.185 |
| 246 | qdrant | e5_base_hf | HNSW_ef32 | 22.456 | 22.176 | 22.815 | 0.185 | 22.411 | 22.778 |
| 247 | qdrant | mpnet_multi_base | HNSW_ef32 | 22.545 | 22.383 | 23.083 | 0.212 | 22.432 | 22.906 |
| 248 | weaviate | e5_base | HNSW_limit20 | 22.62 | 22.154 | 22.892 | 0.209 | 22.628 | 22.869 |
| 249 | qdrant | bge_squad | HNSW_ef32 | 22.791 | 22.355 | 23.244 | 0.269 | 22.82 | 23.162 |
| 250 | qdrant | multi_qa_mpnet | HNSW_ef32 | 23.034 | 22.897 | 23.289 | 0.116 | 23.028 | 23.216 |
| 251 | milvus | snowflake_arctic_l | HNSW_batch | 23.109 | 21.915 | 24.79 | 0.838 | 23.03 | 24.435 |
| 252 | qdrant | snowflake_arctic_l | HNSW_ef32 | 23.441 | 22.933 | 24.128 | 0.387 | 23.442 | 24.003 |
| 253 | milvus | bge_squad | HNSW_batch | 23.598 | 21.288 | 29.519 | 2.096 | 22.943 | 27.189 |
| 254 | qdrant | e5_base | HNSW_ef32 | 24.096 | 23.763 | 24.535 | 0.248 | 23.977 | 24.497 |
| 255 | qdrant | mpnet_base | HNSW_default | 24.734 | 24.29 | 26.424 | 0.649 | 24.502 | 25.993 |
| 256 | qdrant | mpnet_multi | HNSW_default | 24.96 | 24.541 | 25.536 | 0.343 | 24.89 | 25.511 |
| 257 | qdrant | distilroberta | HNSW_default | 24.96 | 24.018 | 26.747 | 0.757 | 24.852 | 26.17 |
| 258 | qdrant | mpnet_base | HNSW_ef128 | 25.044 | 24.787 | 25.749 | 0.257 | 24.981 | 25.476 |
| 259 | weaviate | all_mini_l6_base | HNSW_limit50 | 25.32 | 24.193 | 26.365 | 0.728 | 25.27 | 26.264 |
| 260 | qdrant | gte_base | HNSW_default | 25.468 | 25.106 | 26.028 | 0.305 | 25.368 | 25.922 |
| 261 | qdrant | e5_base_base | HNSW_default | 25.612 | 24.966 | 26.237 | 0.352 | 25.566 | 26.164 |
| 262 | qdrant | gte_base | HNSW_ef128 | 25.81 | 24.924 | 26.729 | 0.743 | 25.908 | 26.708 |
| 263 | qdrant | bge_base_en | HNSW_default | 25.956 | 25.162 | 26.849 | 0.487 | 25.878 | 26.777 |
| 264 | qdrant | distilroberta | HNSW_ef128 | 25.997 | 25.441 | 26.465 | 0.324 | 26.067 | 26.444 |
| 265 | weaviate | minilm_l12 | HNSW_limit50 | 26.127 | 25.217 | 28.169 | 0.764 | 26.028 | 27.337 |
| 266 | qdrant | mpnet_multi | HNSW_ef128 | 26.142 | 25.212 | 27.083 | 0.534 | 26.237 | 26.922 |
| 267 | qdrant | multi_qa_mpnet | HNSW_default | 26.282 | 26.029 | 26.563 | 0.142 | 26.281 | 26.484 |
| 268 | qdrant | mpnet_multi_base | HNSW_default | 26.513 | 26.15 | 26.982 | 0.237 | 26.43 | 26.903 |
| 269 | qdrant | bge_base_en | HNSW_ef128 | 26.523 | 25.907 | 27.491 | 0.467 | 26.416 | 27.273 |
| 270 | weaviate | minilm_l12 | HYBRID_alpha0.5 | 26.57 | 25.874 | 27.801 | 0.531 | 26.383 | 27.439 |
| 271 | weaviate | all_mini_l6_base | HYBRID_alpha0.5 | 26.666 | 26.402 | 27.428 | 0.296 | 26.536 | 27.219 |
| 272 | weaviate | minilm_l12 | HYBRID_alpha0.75 | 26.7 | 26.213 | 27.732 | 0.439 | 26.513 | 27.436 |
| 273 | weaviate | all_mini_l6_base | HYBRID_alpha0.75 | 26.751 | 26.4 | 27.537 | 0.331 | 26.602 | 27.312 |
| 274 | qdrant | e5_base_base | HNSW_ef128 | 26.913 | 26.04 | 27.928 | 0.52 | 26.928 | 27.728 |
| 275 | qdrant | bge_squad_base | HNSW_default | 27.009 | 26.784 | 27.551 | 0.227 | 26.943 | 27.418 |
| 276 | qdrant | mpnet_multi_base | HNSW_ef128 | 27.466 | 26.932 | 28.522 | 0.464 | 27.364 | 28.203 |
| 277 | weaviate | all_mini_l6_base | HYBRID_alpha0.25 | 27.518 | 26.494 | 29.899 | 1.237 | 26.775 | 29.558 |
| 278 | weaviate | mpnet_base | HNSW_limit50 | 27.521 | 27.064 | 28.974 | 0.541 | 27.319 | 28.463 |
| 279 | weaviate | paraphrase_multi | HNSW_limit50 | 27.546 | 26.815 | 28.157 | 0.383 | 27.605 | 28.055 |
| 280 | qdrant | snowflake_arctic_l_base | HNSW_default | 27.681 | 26.452 | 30.162 | 1.041 | 27.527 | 29.485 |
| 281 | qdrant | multi_qa_mpnet | HNSW_ef128 | 27.778 | 27.173 | 28.359 | 0.375 | 27.78 | 28.291 |
| 282 | qdrant | bge_squad | HNSW_default | 27.831 | 27.548 | 28.211 | 0.209 | 27.85 | 28.117 |
| 283 | weaviate | minilm_l12 | HYBRID_alpha0.25 | 27.917 | 26.5 | 29.68 | 1.133 | 27.738 | 29.666 |
| 284 | weaviate | bge_small_en | HNSW_limit50 | 28.243 | 27.617 | 28.9 | 0.37 | 28.278 | 28.753 |
| 285 | weaviate | distilroberta | HNSW_limit50 | 28.244 | 27.454 | 29.181 | 0.501 | 28.163 | 28.992 |
| 286 | weaviate | gte_small | HNSW_limit50 | 28.439 | 27.961 | 28.881 | 0.297 | 28.416 | 28.851 |
| 287 | qdrant | bge_squad_base | HNSW_ef128 | 28.563 | 28.22 | 28.965 | 0.252 | 28.556 | 28.904 |
| 288 | weaviate | paraphrase_multi | HYBRID_alpha0.5 | 28.57 | 28.317 | 28.859 | 0.153 | 28.558 | 28.832 |
| 289 | weaviate | paraphrase_multi | HYBRID_alpha0.75 | 28.595 | 28.259 | 29.209 | 0.273 | 28.535 | 29.104 |
| 290 | weaviate | mpnet_base | HYBRID_alpha0.5 | 28.704 | 28.247 | 29.474 | 0.38 | 28.634 | 29.351 |
| 291 | qdrant | e5_base_hf | HNSW_ef128 | 28.778 | 28.447 | 29.174 | 0.22 | 28.779 | 29.141 |
| 292 | weaviate | mpnet_base | HYBRID_alpha0.75 | 28.818 | 28.142 | 29.794 | 0.454 | 28.845 | 29.487 |
| 293 | qdrant | mpnet_base | HNSW_ef256 | 28.84 | 28.59 | 29.081 | 0.16 | 28.822 | 29.081 |
| 294 | qdrant | snowflake_arctic_l_base | HNSW_ef128 | 28.886 | 28.027 | 31.112 | 0.926 | 28.542 | 30.584 |
| 295 | weaviate | paraphrase_multi | HYBRID_alpha0.25 | 29.177 | 28.341 | 30.46 | 0.751 | 28.954 | 30.352 |
| 296 | weaviate | bge_base_en | HNSW_limit50 | 29.258 | 28.459 | 31.672 | 0.854 | 29.076 | 30.629 |
| 297 | weaviate | bge_small_en | HYBRID_alpha0.5 | 29.476 | 29.181 | 29.761 | 0.187 | 29.528 | 29.738 |
| 298 | qdrant | snowflake_arctic_l | HNSW_default | 29.649 | 29.212 | 30.037 | 0.253 | 29.703 | 29.959 |
| 299 | qdrant | e5_base | HNSW_default | 29.695 | 29.017 | 30.24 | 0.414 | 29.614 | 30.239 |
| 300 | weaviate | multi_qa_minilm | HNSW_limit50 | 29.803 | 29.308 | 30.164 | 0.252 | 29.859 | 30.079 |
| 301 | weaviate | bge_small_en | HYBRID_alpha0.75 | 29.867 | 28.973 | 34.386 | 1.529 | 29.377 | 32.337 |
| 302 | qdrant | bge_squad | HNSW_ef128 | 29.876 | 29.178 | 32.326 | 0.901 | 29.546 | 31.487 |
| 303 | weaviate | mpnet_base | HYBRID_alpha0.25 | 30.043 | 28.391 | 34.151 | 1.843 | 29.326 | 33.336 |
| 304 | weaviate | distilroberta | HYBRID_alpha0.75 | 30.121 | 29.446 | 31.894 | 0.71 | 29.814 | 31.344 |
| 305 | weaviate | distilroberta | HYBRID_alpha0.5 | 30.195 | 29.233 | 31.148 | 0.612 | 30.334 | 31.022 |
| 306 | qdrant | distilroberta | HNSW_ef256 | 30.3 | 29.686 | 31.375 | 0.528 | 30.063 | 31.231 |
| 307 | qdrant | paraphrase_multi | HNSW_ef128 | 30.412 | 29.947 | 30.917 | 0.329 | 30.375 | 30.916 |
| 308 | qdrant | paraphrase_multi | HNSW_ef256 | 30.436 | 30.092 | 31.088 | 0.28 | 30.421 | 30.864 |
| 309 | weaviate | gte_small | HYBRID_alpha0.5 | 30.571 | 29.827 | 30.935 | 0.31 | 30.609 | 30.895 |
| 310 | weaviate | bge_small_en | HYBRID_alpha0.25 | 30.666 | 29.305 | 32.092 | 0.927 | 30.508 | 32.005 |
| 311 | qdrant | paraphrase_multi | HNSW_ef32 | 30.885 | 30.495 | 31.922 | 0.407 | 30.729 | 31.613 |
| 312 | weaviate | multi_qa_minilm | HYBRID_alpha0.75 | 30.919 | 30.609 | 31.116 | 0.164 | 30.996 | 31.088 |
| 313 | weaviate | multi_qa_minilm | HYBRID_alpha0.5 | 31.056 | 30.78 | 31.344 | 0.183 | 30.991 | 31.334 |
| 314 | qdrant | bge_base_en | HNSW_ef256 | 31.075 | 30.623 | 31.603 | 0.306 | 31.133 | 31.484 |
| 315 | qdrant | bge_small_en | HNSW_ef256 | 31.12 | 30.769 | 31.983 | 0.36 | 31.013 | 31.747 |
| 316 | qdrant | all_mini_l6 | HNSW_default | 31.183 | 30.084 | 31.989 | 0.56 | 31.236 | 31.942 |
| 317 | qdrant | minilm_l12 | HNSW_ef128 | 31.262 | 30.685 | 31.856 | 0.376 | 31.261 | 31.81 |
| 318 | weaviate | e5_small_hf | HNSW_limit50 | 31.311 | 30.487 | 31.706 | 0.34 | 31.362 | 31.68 |
| 319 | weaviate | e5_small_base | HNSW_limit50 | 31.318 | 30.312 | 33.088 | 0.792 | 31.256 | 32.655 |
| 320 | qdrant | minilm_l12 | HNSW_ef256 | 31.329 | 30.989 | 32.325 | 0.385 | 31.234 | 32.002 |
| 321 | qdrant | paraphrase_multi | EXACT | 31.356 | 30.602 | 31.858 | 0.429 | 31.435 | 31.853 |
| 322 | qdrant | e5_small_base | HNSW_ef256 | 31.407 | 31.108 | 31.986 | 0.288 | 31.326 | 31.836 |
| 323 | qdrant | e5_small | HNSW_ef32 | 31.472 | 29.941 | 32.703 | 0.873 | 31.434 | 32.667 |
| 324 | qdrant | minilm_l12 | HNSW_ef32 | 31.522 | 30.843 | 32.118 | 0.426 | 31.573 | 32.04 |
| 325 | weaviate | gte_small | HYBRID_alpha0.25 | 31.625 | 30.467 | 33.583 | 1.094 | 31.217 | 33.527 |
| 326 | qdrant | bge_small_en | HNSW_ef128 | 31.662 | 30.978 | 32.227 | 0.365 | 31.636 | 32.216 |
| 327 | weaviate | gte_small | HYBRID_alpha0.75 | 31.666 | 31.073 | 32.273 | 0.36 | 31.669 | 32.196 |
| 328 | qdrant | gte_small | EXACT | 31.688 | 30.596 | 33.288 | 0.732 | 31.586 | 32.842 |
| 329 | qdrant | e5_small_hf | HNSW_ef32 | 31.691 | 30.758 | 32.745 | 0.624 | 31.551 | 32.614 |
| 330 | qdrant | paraphrase_multi | HNSW_default | 31.698 | 30.72 | 32.518 | 0.573 | 31.735 | 32.498 |
| 331 | qdrant | mpnet_multi | HNSW_ef256 | 31.713 | 30.803 | 32.537 | 0.516 | 31.688 | 32.433 |
| 332 | qdrant | bge_small_en | HNSW_default | 31.744 | 30.894 | 32.89 | 0.56 | 31.61 | 32.751 |
| 333 | weaviate | bge_base_en | HYBRID_alpha0.25 | 31.752 | 31.305 | 32.693 | 0.485 | 31.518 | 32.646 |
| 334 | qdrant | gte_small | HNSW_ef128 | 31.79 | 30.66 | 33.076 | 0.77 | 31.925 | 32.818 |
| 335 | weaviate | bge_base_en | HYBRID_alpha0.5 | 31.793 | 31.101 | 32.809 | 0.579 | 31.598 | 32.761 |
| 336 | qdrant | gte_small | HNSW_ef32 | 31.795 | 31.23 | 32.014 | 0.216 | 31.82 | 32.008 |
| 337 | qdrant | e5_base | HNSW_ef128 | 31.869 | 31.281 | 32.513 | 0.344 | 31.814 | 32.346 |
| 338 | qdrant | gte_small | HNSW_default | 31.875 | 31.372 | 33.021 | 0.479 | 31.84 | 32.641 |
| 339 | qdrant | all_mini_l6_base | EXACT | 31.891 | 31.491 | 32.163 | 0.183 | 31.9 | 32.121 |
| 340 | qdrant | bge_small_en | HNSW_ef32 | 31.906 | 30.994 | 33.387 | 0.712 | 31.825 | 33.021 |
| 341 | qdrant | multi_qa_minilm | HNSW_default | 31.968 | 31.136 | 32.928 | 0.635 | 31.941 | 32.811 |
| 342 | qdrant | e5_small | HNSW_ef128 | 31.982 | 30.243 | 33.292 | 0.922 | 32.055 | 33.193 |
| 343 | weaviate | distilroberta | HYBRID_alpha0.25 | 32.036 | 31.172 | 32.702 | 0.456 | 32.024 | 32.656 |
| 344 | qdrant | e5_base_base | HNSW_ef256 | 32.057 | 30.943 | 34.153 | 0.905 | 31.842 | 33.578 |
| 345 | weaviate | mpnet_multi_base | HNSW_limit50 | 32.149 | 31.161 | 33.264 | 0.602 | 32.031 | 33.22 |
| 346 | qdrant | bge_small_en | EXACT | 32.152 | 31.834 | 32.694 | 0.267 | 32.074 | 32.551 |
| 347 | qdrant | all_mini_l6 | HNSW_ef32 | 32.249 | 31.525 | 33.41 | 0.596 | 32.325 | 33.153 |
| 348 | qdrant | all_mini_l6_base | HNSW_ef256 | 32.274 | 31.403 | 33.961 | 0.647 | 32.072 | 33.372 |
| 349 | qdrant | all_mini_l6 | EXACT | 32.291 | 31.387 | 33.292 | 0.491 | 32.275 | 33.055 |
| 350 | qdrant | snowflake_arctic_l | HNSW_ef128 | 32.302 | 31.059 | 34.553 | 0.847 | 32.149 | 33.683 |
| 351 | qdrant | multi_qa_minilm | EXACT | 32.346 | 31.287 | 34.062 | 0.772 | 32.187 | 33.718 |
| 352 | qdrant | mpnet_multi_base | HNSW_ef256 | 32.355 | 32.119 | 32.642 | 0.146 | 32.374 | 32.557 |
| 353 | qdrant | minilm_l12 | HNSW_default | 32.357 | 31.446 | 33.39 | 0.535 | 32.419 | 33.16 |
| 354 | qdrant | all_mini_l6_base | HNSW_ef32 | 32.369 | 31.83 | 33.046 | 0.406 | 32.229 | 33.005 |
| 355 | qdrant | minilm_l12 | EXACT | 32.396 | 31.228 | 33.783 | 0.755 | 32.255 | 33.564 |
| 356 | qdrant | e5_small_hf | HNSW_default | 32.402 | 31.901 | 33.252 | 0.457 | 32.185 | 33.226 |
| 357 | weaviate | multi_qa_minilm | HYBRID_alpha0.25 | 32.409 | 31.048 | 34.305 | 1.221 | 32.055 | 34.227 |
| 358 | qdrant | multi_qa_minilm | HNSW_ef128 | 32.434 | 31.431 | 33.846 | 0.752 | 32.494 | 33.551 |
| 359 | qdrant | gte_small | HNSW_ef256 | 32.468 | 31.038 | 33.533 | 0.788 | 32.583 | 33.477 |
| 360 | qdrant | e5_small_base | EXACT | 32.478 | 31.455 | 33.838 | 0.885 | 32.104 | 33.809 |
| 361 | qdrant | all_mini_l6_base | HNSW_ef128 | 32.494 | 31.749 | 33.236 | 0.481 | 32.49 | 33.146 |
| 362 | qdrant | e5_small_hf | HNSW_ef128 | 32.537 | 31.785 | 33.213 | 0.42 | 32.557 | 33.133 |
| 363 | qdrant | gte_base | HNSW_ef256 | 32.549 | 30.969 | 33.496 | 0.764 | 32.742 | 33.409 |
| 364 | qdrant | e5_small_hf | EXACT | 32.552 | 31.916 | 33.679 | 0.46 | 32.541 | 33.284 |
| 365 | qdrant | e5_small_base | HNSW_ef32 | 32.597 | 31.535 | 37.91 | 1.828 | 31.931 | 35.647 |
| 366 | qdrant | all_mini_l6_base | HNSW_default | 32.597 | 31.773 | 33.151 | 0.404 | 32.569 | 33.141 |
| 367 | qdrant | multi_qa_minilm | HNSW_ef32 | 32.614 | 31.415 | 33.568 | 0.637 | 32.877 | 33.318 |
| 368 | qdrant | e5_small | HNSW_ef256 | 32.675 | 31.698 | 34.559 | 0.71 | 32.618 | 33.794 |
| 369 | qdrant | e5_small | EXACT | 32.728 | 32.001 | 33.623 | 0.484 | 32.744 | 33.403 |
| 370 | weaviate | gte_base | HNSW_limit50 | 32.736 | 31.978 | 35.621 | 1.016 | 32.434 | 34.449 |
| 371 | qdrant | all_mini_l6 | HNSW_ef128 | 32.785 | 30.279 | 37.331 | 1.749 | 32.636 | 35.514 |
| 372 | qdrant | e5_small_base | HNSW_ef128 | 32.789 | 31.429 | 36.229 | 1.268 | 32.754 | 34.791 |
| 373 | qdrant | e5_small_hf | HNSW_ef256 | 32.958 | 31.811 | 36.082 | 1.123 | 32.684 | 34.817 |
| 374 | qdrant | all_mini_l6 | HNSW_ef256 | 33.112 | 31.788 | 34.573 | 0.815 | 32.994 | 34.336 |
| 375 | qdrant | multi_qa_minilm | HNSW_ef256 | 33.123 | 32.783 | 33.683 | 0.305 | 33.015 | 33.613 |
| 376 | qdrant | multi_qa_mpnet | HNSW_ef256 | 33.254 | 32.888 | 33.742 | 0.238 | 33.217 | 33.632 |
| 377 | weaviate | bge_base_en | HYBRID_alpha0.75 | 33.264 | 32.108 | 34.727 | 0.855 | 33.112 | 34.531 |
| 378 | weaviate | multi_qa_mpnet | HNSW_limit50 | 33.308 | 32.725 | 33.767 | 0.351 | 33.295 | 33.762 |
| 379 | weaviate | mpnet_multi_base | HYBRID_alpha0.75 | 33.443 | 32.856 | 34.622 | 0.484 | 33.345 | 34.269 |
| 380 | weaviate | bge_squad_base | HNSW_limit50 | 33.473 | 32.827 | 34.364 | 0.44 | 33.443 | 34.232 |
| 381 | qdrant | e5_small | HNSW_default | 33.634 | 32.853 | 34.76 | 0.526 | 33.508 | 34.501 |
| 382 | weaviate | e5_small_base | HYBRID_alpha0.5 | 33.719 | 33.139 | 34.972 | 0.679 | 33.355 | 34.904 |
| 383 | qdrant | e5_small_base | HNSW_default | 33.974 | 32.049 | 38.39 | 1.642 | 33.694 | 36.775 |
| 384 | qdrant | e5_base_hf | HNSW_ef256 | 34.041 | 31.623 | 36.09 | 1.537 | 34.647 | 35.733 |
| 385 | weaviate | e5_small_hf | HYBRID_alpha0.25 | 34.409 | 33.897 | 35.08 | 0.376 | 34.429 | 34.976 |
| 386 | weaviate | e5_small_base | HYBRID_alpha0.25 | 34.543 | 32.99 | 36.887 | 1.457 | 33.688 | 36.798 |
| 387 | weaviate | gte_base | HYBRID_alpha0.75 | 34.585 | 34.274 | 34.934 | 0.215 | 34.604 | 34.879 |
| 388 | weaviate | e5_small_base | HYBRID_alpha0.75 | 34.742 | 33.164 | 37.207 | 1.364 | 34.233 | 36.98 |
| 389 | qdrant | bge_squad_base | HNSW_ef256 | 34.757 | 34.45 | 35.282 | 0.277 | 34.64 | 35.232 |
| 390 | weaviate | snowflake_arctic_l_base | HNSW_limit50 | 34.82 | 33.916 | 35.535 | 0.455 | 34.812 | 35.438 |
| 391 | weaviate | gte_base | HYBRID_alpha0.25 | 34.868 | 34.494 | 35.256 | 0.249 | 34.83 | 35.201 |
| 392 | weaviate | mpnet_multi | HNSW_limit50 | 35.248 | 33.454 | 36.379 | 0.863 | 35.48 | 36.298 |
| 393 | qdrant | snowflake_arctic_l_base | HNSW_ef256 | 35.355 | 34.539 | 37.194 | 0.798 | 35.091 | 36.688 |
| 394 | weaviate | all_mini_l6 | HNSW_limit50 | 35.507 | 34.57 | 37.702 | 0.78 | 35.311 | 36.794 |
| 395 | weaviate | gte_base | HYBRID_alpha0.5 | 35.559 | 34.555 | 36.579 | 0.754 | 35.601 | 36.532 |
| 396 | weaviate | mpnet_multi_base | HYBRID_alpha0.5 | 35.626 | 33.633 | 37.063 | 1.136 | 36.016 | 36.895 |
| 397 | weaviate | e5_base_hf | HNSW_limit50 | 35.891 | 35.128 | 36.695 | 0.474 | 35.86 | 36.587 |
| 398 | weaviate | e5_base_base | HNSW_limit50 | 35.917 | 34.305 | 41.272 | 1.929 | 35.424 | 39.267 |
| 399 | weaviate | multi_qa_mpnet | HYBRID_alpha0.75 | 36.22 | 35.218 | 37.194 | 0.553 | 36.392 | 36.963 |
| 400 | weaviate | multi_qa_mpnet | HYBRID_alpha0.5 | 36.421 | 34.978 | 37.676 | 0.962 | 36.464 | 37.642 |
| 401 | weaviate | e5_small_hf | HYBRID_alpha0.75 | 36.433 | 35.15 | 37.359 | 0.768 | 36.528 | 37.31 |
| 402 | qdrant | bge_squad | HNSW_ef256 | 36.507 | 35.758 | 37.868 | 0.71 | 36.278 | 37.766 |
| 403 | weaviate | e5_small | HNSW_limit50 | 37.064 | 36.252 | 38.149 | 0.692 | 36.794 | 38.11 |
| 404 | weaviate | multi_qa_mpnet | HYBRID_alpha0.25 | 37.118 | 36.341 | 38.742 | 0.724 | 36.885 | 38.358 |
| 405 | weaviate | bge_squad | HNSW_limit50 | 37.128 | 35.599 | 39.042 | 1.024 | 37.009 | 38.663 |
| 406 | weaviate | bge_squad_base | HYBRID_alpha0.5 | 37.673 | 36.174 | 39.143 | 0.816 | 37.924 | 38.705 |
| 407 | weaviate | e5_small_hf | HYBRID_alpha0.5 | 38.625 | 33.792 | 69.456 | 10.431 | 34.391 | 55.74 |
| 408 | weaviate | bge_squad_base | HYBRID_alpha0.75 | 38.76 | 37.36 | 42.235 | 1.361 | 38.363 | 41.252 |
| 409 | weaviate | bge_squad_base | HYBRID_alpha0.25 | 38.87 | 38.089 | 39.909 | 0.578 | 39.041 | 39.666 |
| 410 | weaviate | snowflake_arctic_l_base | HYBRID_alpha0.75 | 38.924 | 37.623 | 41.172 | 0.959 | 38.632 | 40.628 |
| 411 | weaviate | e5_base_base | HYBRID_alpha0.75 | 39.047 | 38.091 | 40.094 | 0.695 | 39.101 | 40.059 |
| 412 | weaviate | snowflake_arctic_l | HNSW_limit50 | 39.363 | 37.941 | 41.126 | 0.847 | 39.289 | 40.799 |
| 413 | weaviate | snowflake_arctic_l_base | HYBRID_alpha0.5 | 39.412 | 38.095 | 40.462 | 0.818 | 39.742 | 40.402 |
| 414 | weaviate | snowflake_arctic_l_base | HYBRID_alpha0.25 | 39.938 | 38.535 | 40.827 | 0.764 | 40.041 | 40.821 |
| 415 | weaviate | mpnet_multi | HYBRID_alpha0.5 | 39.983 | 38.121 | 45.227 | 2.124 | 39.098 | 43.72 |
| 416 | weaviate | all_mini_l6 | HYBRID_alpha0.75 | 40.029 | 38.183 | 43.741 | 1.499 | 39.745 | 42.517 |
| 417 | weaviate | all_mini_l6 | HYBRID_alpha0.25 | 40.132 | 38.514 | 41.321 | 0.799 | 40.167 | 41.292 |
| 418 | weaviate | mpnet_multi | HYBRID_alpha0.75 | 40.167 | 38.654 | 42.69 | 1.192 | 40.03 | 42.1 |
| 419 | weaviate | mpnet_multi | HYBRID_alpha0.25 | 40.252 | 37.699 | 44.777 | 2.329 | 39.046 | 44.359 |
| 420 | weaviate | e5_base_hf | HYBRID_alpha0.25 | 40.288 | 38.287 | 41.511 | 1.14 | 40.649 | 41.496 |
| 421 | qdrant | e5_base | HNSW_ef256 | 40.408 | 38.106 | 42.114 | 1.371 | 40.633 | 41.975 |
| 422 | weaviate | e5_base_base | HYBRID_alpha0.5 | 40.882 | 39.925 | 42.832 | 0.775 | 40.818 | 42.163 |
| 423 | weaviate | mpnet_multi_base | HYBRID_alpha0.25 | 40.892 | 34.092 | 69.254 | 10.108 | 37.571 | 59.494 |
| 424 | weaviate | e5_base_hf | HYBRID_alpha0.5 | 40.962 | 39.943 | 41.695 | 0.507 | 40.968 | 41.6 |
| 425 | weaviate | e5_base_hf | HYBRID_alpha0.75 | 40.995 | 40.232 | 42.225 | 0.662 | 40.923 | 42.117 |
| 426 | qdrant | snowflake_arctic_l | HNSW_ef256 | 41.072 | 39.716 | 42.756 | 1.029 | 41.191 | 42.42 |
| 427 | weaviate | e5_base_base | HYBRID_alpha0.25 | 41.47 | 39.834 | 43.149 | 0.943 | 41.73 | 42.758 |
| 428 | weaviate | e5_small | HYBRID_alpha0.25 | 41.71 | 40.555 | 43.289 | 0.748 | 41.861 | 42.759 |
| 429 | weaviate | bge_squad | HYBRID_alpha0.25 | 42.267 | 40.73 | 43.745 | 0.871 | 42.229 | 43.551 |
| 430 | weaviate | bge_squad | HYBRID_alpha0.75 | 42.6 | 39.922 | 44.407 | 1.478 | 42.876 | 44.383 |
| 431 | weaviate | bge_squad | HYBRID_alpha0.5 | 42.629 | 41.056 | 44.503 | 1.002 | 42.742 | 44.204 |
| 432 | weaviate | all_mini_l6 | HYBRID_alpha0.5 | 43.739 | 39.053 | 73.929 | 10.102 | 40.094 | 59.513 |
| 433 | weaviate | e5_base | HNSW_limit50 | 44.04 | 42.104 | 47.065 | 1.638 | 43.755 | 46.635 |
| 434 | weaviate | snowflake_arctic_l | HYBRID_alpha0.5 | 46.375 | 44.849 | 48.316 | 1.182 | 46.288 | 48.303 |
| 435 | weaviate | snowflake_arctic_l | HYBRID_alpha0.25 | 46.394 | 44.888 | 47.702 | 0.827 | 46.476 | 47.578 |
| 436 | weaviate | snowflake_arctic_l | HYBRID_alpha0.75 | 46.721 | 45.889 | 47.683 | 0.587 | 46.621 | 47.633 |
| 437 | qdrant | e5_base_hf | HNSW_default | 49.591 | 26.727 | 165.641 | 46.04 | 27.395 | 141.105 |
| 438 | weaviate | e5_base | HYBRID_alpha0.25 | 49.605 | 48.602 | 51.089 | 0.74 | 49.548 | 50.846 |
| 439 | weaviate | e5_base | HYBRID_alpha0.75 | 49.934 | 47.39 | 55.537 | 2.146 | 49.923 | 53.259 |
| 440 | weaviate | e5_base | HYBRID_alpha0.5 | 50.314 | 48.226 | 54.272 | 1.464 | 50.149 | 52.634 |
| 441 | weaviate | e5_small | HYBRID_alpha0.5 | 50.823 | 42.31 | 93.092 | 15.105 | 44.547 | 79.038 |
| 442 | qdrant | e5_base | EXACT | 55.686 | 53.683 | 57.694 | 1.001 | 55.67 | 57.156 |
| 443 | qdrant | mpnet_multi_base | EXACT | 56.143 | 55.196 | 57.606 | 0.837 | 55.805 | 57.419 |
| 444 | qdrant | e5_base_hf | EXACT | 56.352 | 54.476 | 57.719 | 0.878 | 56.279 | 57.686 |
| 445 | qdrant | mpnet_multi | EXACT | 56.456 | 52.558 | 60.754 | 2.331 | 56.081 | 59.837 |
| 446 | qdrant | e5_base_base | EXACT | 57.747 | 55.466 | 59.98 | 1.572 | 57.867 | 59.837 |
| 447 | qdrant | distilroberta | EXACT | 57.878 | 55.424 | 63.529 | 2.164 | 57.843 | 61.309 |
| 448 | qdrant | bge_base_en | EXACT | 57.943 | 56.22 | 60.154 | 1.084 | 57.77 | 59.574 |
| 449 | qdrant | mpnet_base | EXACT | 57.963 | 56.334 | 59.554 | 1.119 | 57.917 | 59.547 |
| 450 | qdrant | multi_qa_mpnet | EXACT | 57.99 | 56.508 | 59.508 | 1.023 | 57.764 | 59.37 |
| 451 | qdrant | gte_base | EXACT | 58.965 | 56.935 | 63.967 | 1.94 | 58.667 | 62.244 |
| 452 | weaviate | e5_small | HYBRID_alpha0.75 | 59.726 | 40.007 | 90.063 | 21.624 | 47.118 | 89.715 |
| 453 | qdrant | snowflake_arctic_l_base | EXACT | 69.713 | 68.534 | 71.533 | 0.826 | 69.596 | 71.148 |
| 454 | qdrant | bge_squad | EXACT | 73.479 | 69.238 | 75.751 | 1.907 | 73.933 | 75.746 |
| 455 | qdrant | bge_squad_base | EXACT | 73.632 | 71.604 | 76.404 | 1.497 | 73.443 | 76.198 |
| 456 | qdrant | snowflake_arctic_l | EXACT | 74.538 | 71.425 | 77.361 | 1.807 | 74.879 | 76.826 |
| 457 | milvus | bge_small_en | HNSW_limit20 | 75.179 | 74.011 | 77.347 | 1.056 | 74.964 | 77.086 |
| 458 | milvus | paraphrase_multi | HNSW_limit20 | 76.441 | 74.93 | 80.118 | 1.587 | 76.195 | 79.01 |
| 459 | milvus | bge_small_en | HNSW_limit50 | 76.886 | 75.141 | 79.404 | 1.291 | 76.753 | 78.732 |
| 460 | milvus | paraphrase_multi | HNSW_limit5 | 77.187 | 75.722 | 78.39 | 0.853 | 77.455 | 78.225 |
| 461 | milvus | bge_small_en | HNSW_limit5 | 77.276 | 74.132 | 85.302 | 3.219 | 76.795 | 82.539 |
| 462 | milvus | all_mini_l6 | HNSW_limit5 | 77.302 | 76.045 | 78.545 | 0.822 | 77.364 | 78.408 |
| 463 | milvus | paraphrase_multi | HNSW_default | 77.33 | 76.578 | 78.207 | 0.613 | 77.131 | 78.204 |
| 464 | milvus | gte_small | HNSW_default | 77.519 | 76.056 | 78.591 | 0.882 | 77.65 | 78.585 |
| 465 | milvus | e5_small_hf | HNSW_default | 77.689 | 75.228 | 80.815 | 1.644 | 77.51 | 80.339 |
| 466 | milvus | all_mini_l6_base | HNSW_limit20 | 77.704 | 77.315 | 78.355 | 0.353 | 77.595 | 78.342 |
| 467 | milvus | e5_small_base | HNSW_default | 77.734 | 77.104 | 78.24 | 0.419 | 77.736 | 78.217 |
| 468 | milvus | paraphrase_multi | HNSW_limit50 | 77.87 | 76.146 | 79.987 | 1.346 | 77.61 | 79.713 |
| 469 | milvus | minilm_l12 | HNSW_limit20 | 78.091 | 77.189 | 78.979 | 0.549 | 78.097 | 78.828 |
| 470 | milvus | e5_small_base | HNSW_limit5 | 78.093 | 77.079 | 79.232 | 0.635 | 78.096 | 79.046 |
| 471 | milvus | gte_small | HNSW_limit20 | 78.116 | 77.046 | 79.981 | 0.934 | 77.747 | 79.601 |
| 472 | milvus | all_mini_l6_base | HNSW_default | 78.183 | 75.807 | 79.402 | 1.193 | 78.3 | 79.356 |
| 473 | milvus | all_mini_l6 | HNSW_default | 78.274 | 77.636 | 78.868 | 0.307 | 78.252 | 78.719 |
| 474 | milvus | e5_small_hf | HNSW_limit5 | 78.445 | 76.814 | 79.729 | 0.955 | 78.55 | 79.596 |
| 475 | milvus | e5_small_hf | HNSW_limit20 | 78.462 | 77.54 | 79.526 | 0.7 | 78.294 | 79.513 |
| 476 | milvus | all_mini_l6 | HNSW_limit20 | 78.615 | 77.513 | 79.359 | 0.567 | 78.655 | 79.312 |
| 477 | milvus | gte_small | HNSW_limit50 | 78.772 | 77.632 | 80.779 | 0.805 | 78.68 | 80.115 |
| 478 | milvus | multi_qa_minilm | HNSW_limit20 | 78.782 | 76.099 | 87.592 | 3.789 | 76.684 | 86.25 |
| 479 | milvus | e5_small | HNSW_limit5 | 78.843 | 78.231 | 79.796 | 0.505 | 78.563 | 79.633 |
| 480 | milvus | all_mini_l6 | HNSW_limit50 | 78.927 | 78.335 | 80.04 | 0.45 | 78.878 | 79.69 |
| 481 | milvus | gte_small | HNSW_limit5 | 79.105 | 76.027 | 84.653 | 2.983 | 77.991 | 84.498 |
| 482 | milvus | minilm_l12 | HNSW_limit5 | 79.176 | 75.348 | 87.481 | 3.744 | 77.212 | 85.883 |
| 483 | milvus | bge_small_en | HNSW_default | 79.196 | 75.778 | 87.096 | 3.038 | 78.541 | 84.303 |
| 484 | milvus | minilm_l12 | HNSW_default | 79.236 | 77.986 | 80.858 | 1.032 | 78.892 | 80.757 |
| 485 | milvus | e5_small | HNSW_limit20 | 79.278 | 78.268 | 80.996 | 0.824 | 79.111 | 80.697 |
| 486 | milvus | multi_qa_minilm | HNSW_limit5 | 79.334 | 75.447 | 86.174 | 2.912 | 78.565 | 84.363 |
| 487 | milvus | multi_qa_minilm | HNSW_default | 79.338 | 77.255 | 83.029 | 1.816 | 78.699 | 82.591 |
| 488 | milvus | all_mini_l6_base | HNSW_limit50 | 79.673 | 78.604 | 80.461 | 0.515 | 79.695 | 80.373 |
| 489 | milvus | e5_small_hf | HNSW_limit50 | 79.833 | 78.714 | 83.363 | 1.354 | 79.453 | 82.157 |
| 490 | milvus | all_mini_l6_base | HNSW_limit5 | 79.883 | 77.151 | 82.118 | 1.401 | 79.899 | 81.656 |
| 491 | milvus | multi_qa_minilm | HNSW_limit50 | 80.211 | 79.395 | 80.981 | 0.58 | 80.14 | 80.921 |
| 492 | milvus | e5_small | HNSW_limit50 | 80.596 | 79.3 | 83.08 | 1.131 | 80.082 | 82.494 |
| 493 | milvus | minilm_l12 | HNSW_limit50 | 80.743 | 78.7 | 82.467 | 1.316 | 81.094 | 82.334 |
| 494 | milvus | e5_small | HNSW_default | 80.814 | 77.619 | 91.886 | 4.035 | 79.454 | 88.183 |
| 495 | milvus | e5_small_base | HNSW_limit50 | 81.179 | 79.255 | 86.623 | 2.327 | 80.202 | 85.767 |
| 496 | milvus | e5_small_base | HNSW_limit20 | 83.749 | 79.457 | 107.386 | 8.882 | 79.653 | 101.066 |
| 497 | milvus | bge_base_en | HNSW_limit20 | 126.964 | 125.79 | 127.928 | 0.656 | 127.134 | 127.784 |
| 498 | milvus | bge_base_en | HNSW_limit5 | 127.629 | 126.458 | 129.278 | 0.813 | 127.569 | 128.906 |
| 499 | milvus | mpnet_multi_base | HNSW_default | 128.168 | 126.85 | 129.388 | 0.793 | 128.031 | 129.276 |
| 500 | milvus | mpnet_multi_base | HNSW_limit5 | 128.227 | 125.75 | 132.398 | 1.689 | 128.077 | 130.949 |
| 501 | milvus | bge_base_en | HNSW_limit50 | 128.469 | 127.333 | 129.28 | 0.564 | 128.488 | 129.197 |
| 502 | milvus | e5_base | HNSW_limit5 | 128.59 | 127.748 | 129.818 | 0.724 | 128.52 | 129.747 |
| 503 | milvus | e5_base | HNSW_limit20 | 128.783 | 126.928 | 131.343 | 1.372 | 128.355 | 131.164 |
| 504 | milvus | e5_base | HNSW_default | 129.113 | 127.671 | 130.211 | 0.837 | 129.232 | 130.187 |
| 505 | milvus | mpnet_multi_base | HNSW_limit20 | 129.123 | 126.988 | 130.945 | 1.275 | 129.227 | 130.841 |
| 506 | milvus | e5_base_base | HNSW_limit5 | 129.156 | 126.284 | 132.541 | 1.911 | 129.856 | 131.649 |
| 507 | milvus | gte_base | HNSW_limit5 | 129.171 | 128.659 | 130.337 | 0.467 | 129.033 | 130.005 |
| 508 | milvus | bge_base_en | HNSW_default | 129.182 | 127.832 | 135.14 | 2.07 | 128.598 | 132.819 |
| 509 | milvus | e5_base_base | HNSW_default | 129.279 | 128.511 | 131.24 | 0.844 | 128.922 | 130.932 |
| 510 | milvus | mpnet_multi_base | HNSW_limit50 | 129.365 | 127.81 | 131.902 | 1.233 | 129.148 | 131.292 |
| 511 | milvus | gte_base | HNSW_default | 129.401 | 128.49 | 130.61 | 0.59 | 129.407 | 130.327 |
| 512 | milvus | gte_base | HNSW_limit20 | 129.624 | 128.451 | 131.378 | 0.841 | 129.514 | 130.976 |
| 513 | milvus | multi_qa_mpnet | HNSW_limit5 | 129.715 | 128.747 | 130.851 | 0.656 | 129.622 | 130.643 |
| 514 | milvus | multi_qa_mpnet | HNSW_limit20 | 129.832 | 129.135 | 130.838 | 0.473 | 129.837 | 130.587 |
| 515 | milvus | multi_qa_mpnet | HNSW_default | 130.164 | 129.022 | 131.296 | 0.625 | 130.097 | 131.096 |
| 516 | milvus | mpnet_multi | HNSW_limit5 | 130.268 | 129.181 | 132.731 | 1.088 | 129.919 | 132.35 |
| 517 | milvus | e5_base | HNSW_limit50 | 130.473 | 128.674 | 131.789 | 1.068 | 130.491 | 131.775 |
| 518 | milvus | e5_base_base | HNSW_limit20 | 130.784 | 129.471 | 132.092 | 0.806 | 130.696 | 131.937 |
| 519 | milvus | distilroberta | HNSW_limit20 | 130.862 | 129.315 | 131.966 | 0.768 | 130.946 | 131.958 |
| 520 | milvus | mpnet_base | HNSW_default | 131.031 | 130.114 | 132.161 | 0.684 | 131.143 | 131.931 |
| 521 | milvus | multi_qa_mpnet | HNSW_limit50 | 131.178 | 130.001 | 133.024 | 0.986 | 130.958 | 132.92 |
| 522 | milvus | e5_base_hf | HNSW_limit5 | 131.252 | 128.936 | 137.868 | 2.402 | 130.555 | 135.591 |
| 523 | milvus | e5_base_hf | HNSW_default | 131.398 | 130.349 | 132.468 | 0.631 | 131.322 | 132.393 |
| 524 | milvus | distilroberta | HNSW_default | 131.517 | 130.12 | 133.646 | 1.142 | 131.338 | 133.429 |
| 525 | milvus | mpnet_multi | HNSW_limit50 | 131.685 | 130.684 | 133.097 | 0.774 | 131.553 | 132.824 |
| 526 | milvus | gte_base | HNSW_limit50 | 131.728 | 130.102 | 134.221 | 1.345 | 131.681 | 133.767 |
| 527 | milvus | mpnet_multi | HNSW_default | 131.752 | 130.132 | 133.379 | 1.099 | 131.502 | 133.33 |
| 528 | milvus | distilroberta | HNSW_limit5 | 131.764 | 130.124 | 133.194 | 1.004 | 131.912 | 133.052 |
| 529 | milvus | mpnet_base | HNSW_limit5 | 131.981 | 129.016 | 143.527 | 3.93 | 130.785 | 138.336 |
| 530 | milvus | e5_base_hf | HNSW_limit20 | 132.236 | 130.405 | 135.018 | 1.458 | 131.787 | 134.591 |
| 531 | milvus | distilroberta | HNSW_limit50 | 132.383 | 131.41 | 133.711 | 0.811 | 132.161 | 133.634 |
| 532 | milvus | e5_base_base | HNSW_limit50 | 132.39 | 130.803 | 134.496 | 0.987 | 132.012 | 133.873 |
| 533 | milvus | e5_base_hf | HNSW_limit50 | 132.462 | 130.05 | 135.764 | 1.809 | 132.066 | 135.217 |
| 534 | milvus | mpnet_base | HNSW_limit50 | 132.601 | 129.849 | 134.11 | 1.463 | 133.02 | 134.098 |
| 535 | milvus | mpnet_multi | HNSW_limit20 | 133.339 | 130.676 | 139.155 | 2.198 | 132.946 | 137.178 |
| 536 | milvus | mpnet_base | HNSW_limit20 | 134.216 | 129.42 | 143.251 | 3.966 | 132.447 | 141.717 |
| 537 | milvus | bge_squad_base | HNSW_limit5 | 159.339 | 156.317 | 161.339 | 1.447 | 159.649 | 161.116 |
| 538 | milvus | bge_squad_base | HNSW_limit20 | 160.19 | 159.224 | 162.289 | 0.941 | 159.904 | 161.818 |
| 539 | milvus | bge_squad_base | HNSW_default | 160.2 | 157.49 | 163.826 | 1.868 | 159.95 | 163.406 |
| 540 | milvus | bge_squad_base | HNSW_limit50 | 160.91 | 158.513 | 162.835 | 1.065 | 161.036 | 162.461 |
| 541 | milvus | snowflake_arctic_l_base | HNSW_limit5 | 161.148 | 159.51 | 162.916 | 0.832 | 161.133 | 162.354 |
| 542 | milvus | snowflake_arctic_l_base | HNSW_limit20 | 162.093 | 160.283 | 164.832 | 1.373 | 161.693 | 164.242 |
| 543 | milvus | snowflake_arctic_l_base | HNSW_default | 162.523 | 160.778 | 164.335 | 1.12 | 162.221 | 164.256 |
| 544 | milvus | snowflake_arctic_l_base | HNSW_limit50 | 164.768 | 163.044 | 167.475 | 1.176 | 164.39 | 166.764 |
| 545 | milvus | bge_squad | HNSW_default | 166.793 | 165.948 | 168.308 | 0.726 | 166.417 | 167.998 |
| 546 | milvus | bge_squad | HNSW_limit50 | 167.385 | 166.062 | 168.452 | 0.795 | 167.355 | 168.43 |
| 547 | milvus | bge_squad | HNSW_limit5 | 167.763 | 165.306 | 175.393 | 2.8 | 167.332 | 172.533 |
| 548 | milvus | bge_squad | HNSW_limit20 | 168.097 | 166.27 | 173.688 | 2.095 | 167.581 | 171.955 |
| 549 | milvus | snowflake_arctic_l | HNSW_limit5 | 170.208 | 169.064 | 171.838 | 1.009 | 169.861 | 171.78 |
| 550 | milvus | snowflake_arctic_l | HNSW_limit20 | 170.418 | 169.473 | 171.901 | 0.78 | 170.389 | 171.586 |
| 551 | milvus | snowflake_arctic_l | HNSW_limit50 | 171.06 | 168.93 | 172.905 | 1.264 | 171.442 | 172.59 |
| 552 | milvus | snowflake_arctic_l | HNSW_default | 173.956 | 168.489 | 187.675 | 5.977 | 171.418 | 185.051 |
| 553 | lancedb | e5_small | VECTOR_default | 451.783 | 447.711 | 457.505 | 2.896 | 451.928 | 456.22 |
| 554 | lancedb | e5_small | VECTOR_limit5 | 458.725 | 448.5 | 504.818 | 15.932 | 453.762 | 486.303 |
| 555 | lancedb | e5_small | VECTOR_cosine | 460.358 | 455.57 | 469.308 | 4.188 | 460.227 | 466.726 |
| 556 | lancedb | multi_qa_minilm | VECTOR_default | 461.814 | 455.355 | 471.286 | 4.242 | 462.338 | 468.237 |
| 557 | lancedb | multi_qa_minilm | VECTOR_cosine | 464.111 | 459.157 | 470.921 | 3.294 | 463.824 | 469.145 |
| 558 | lancedb | all_mini_l6_base | VECTOR_default | 464.431 | 458.239 | 472.446 | 4.284 | 464.057 | 471.2 |
| 559 | lancedb | all_mini_l6_base | VECTOR_limit5 | 464.508 | 456.322 | 510.767 | 15.544 | 459.828 | 488.723 |
| 560 | lancedb | multi_qa_minilm | VECTOR_limit5 | 465.176 | 453.618 | 511.363 | 15.727 | 460.772 | 490.875 |
| 561 | lancedb | all_mini_l6_base | VECTOR_cosine | 465.835 | 463.071 | 470.379 | 2.407 | 465.03 | 469.927 |
| 562 | lancedb | all_mini_l6_base | VECTOR_L2 | 465.848 | 460.47 | 470.801 | 3.467 | 465.514 | 470.723 |
| 563 | lancedb | gte_small | VECTOR_cosine | 466.035 | 460.55 | 471.198 | 3.053 | 466.156 | 470.229 |
| 564 | lancedb | gte_small | VECTOR_default | 467.021 | 459.525 | 476.42 | 4.587 | 466.616 | 473.71 |
| 565 | lancedb | gte_small | VECTOR_L2 | 467.224 | 461.437 | 476.115 | 4.382 | 466.177 | 475.165 |
| 566 | lancedb | e5_small | VECTOR_limit20 | 467.504 | 463.524 | 472.329 | 2.54 | 466.937 | 471.57 |
| 567 | lancedb | multi_qa_minilm | VECTOR_L2 | 467.999 | 465.388 | 473.464 | 2.523 | 466.714 | 472.225 |
| 568 | lancedb | all_mini_l6 | VECTOR_default | 468.509 | 463.061 | 474.968 | 3.722 | 467.934 | 473.852 |
| 569 | lancedb | gte_small | VECTOR_limit5 | 468.712 | 458.667 | 514.023 | 15.46 | 465.142 | 493.494 |
| 570 | lancedb | e5_small_hf | VECTOR_default | 469.366 | 461.673 | 478.909 | 4.846 | 468.96 | 477.416 |
| 571 | lancedb | e5_small_hf | VECTOR_cosine | 469.577 | 465.154 | 473.775 | 3.041 | 469.718 | 473.659 |
| 572 | lancedb | e5_small | VECTOR_L2 | 469.833 | 458.461 | 492.823 | 11.027 | 466.645 | 490.796 |
| 573 | lancedb | paraphrase_multi | VECTOR_limit5 | 469.867 | 460.802 | 514.091 | 14.902 | 465.914 | 493.627 |
| 574 | lancedb | e5_small_base | VECTOR_limit5 | 470.086 | 458.392 | 512.526 | 14.83 | 465.31 | 494.882 |
| 575 | lancedb | paraphrase_multi | VECTOR_cosine | 470.295 | 465.902 | 473.471 | 2.281 | 470.502 | 473.122 |
| 576 | lancedb | paraphrase_multi | VECTOR_default | 471.66 | 461.84 | 510.462 | 13.223 | 467.652 | 493.079 |
| 577 | lancedb | bge_small_en | VECTOR_limit5 | 471.662 | 461.18 | 523.103 | 17.34 | 466.251 | 498.888 |
| 578 | lancedb | all_mini_l6_base | VECTOR_limit20 | 472.198 | 467.993 | 484.15 | 4.387 | 471.48 | 479.579 |
| 579 | lancedb | e5_small_base | VECTOR_default | 472.292 | 465.555 | 484.89 | 5.406 | 470.672 | 481.911 |
| 580 | lancedb | bge_small_en | VECTOR_default | 472.593 | 466.412 | 477.328 | 3.608 | 472.844 | 477.303 |
| 581 | lancedb | bge_small_en | VECTOR_cosine | 472.707 | 463.984 | 480.418 | 4.702 | 472.116 | 480.068 |
| 582 | lancedb | all_mini_l6 | VECTOR_cosine | 472.735 | 469.185 | 476.78 | 2.473 | 472.832 | 476.358 |
| 583 | lancedb | minilm_l12 | VECTOR_L2 | 473.026 | 464.156 | 485.156 | 6.403 | 470.566 | 483.69 |
| 584 | lancedb | minilm_l12 | VECTOR_cosine | 473.038 | 466.718 | 478.238 | 3.17 | 472.408 | 477.545 |
| 585 | lancedb | e5_small_base | VECTOR_cosine | 473.205 | 468.5 | 481.055 | 3.729 | 472.183 | 479.797 |
| 586 | lancedb | e5_small_hf | VECTOR_limit5 | 473.292 | 458.506 | 522.549 | 16.876 | 469.224 | 500.062 |
| 587 | lancedb | e5_small_base | VECTOR_L2 | 473.446 | 466.413 | 479.414 | 3.816 | 473.563 | 479.092 |
| 588 | lancedb | e5_small_hf | VECTOR_L2 | 473.68 | 467.478 | 481.117 | 3.565 | 473.286 | 479.163 |
| 589 | lancedb | paraphrase_multi | VECTOR_L2 | 473.91 | 464.458 | 491.216 | 7.27 | 472.61 | 487.428 |
| 590 | lancedb | minilm_l12 | VECTOR_default | 474.029 | 466.793 | 482.933 | 4.255 | 474.066 | 480.265 |
| 591 | lancedb | all_mini_l6 | VECTOR_L2 | 474.04 | 464.498 | 495.768 | 8.787 | 472.85 | 489.234 |
| 592 | lancedb | multi_qa_minilm | VECTOR_limit20 | 474.162 | 466.016 | 492.217 | 6.945 | 472.588 | 485.641 |
| 593 | lancedb | gte_base | VECTOR_default | 474.25 | 467.869 | 485.756 | 4.622 | 473.186 | 481.833 |
| 594 | lancedb | all_mini_l6 | VECTOR_limit5 | 474.256 | 464.287 | 525.457 | 17.337 | 467.259 | 502.572 |
| 595 | lancedb | e5_base | VECTOR_default | 474.639 | 471.145 | 480.265 | 2.666 | 473.902 | 479.218 |
| 596 | lancedb | minilm_l12 | VECTOR_limit5 | 474.802 | 464.794 | 526.77 | 17.674 | 469.212 | 503.945 |
| 597 | lancedb | paraphrase_multi | VECTOR_limit20 | 475.51 | 471.906 | 478.179 | 2.232 | 476.603 | 477.923 |
| 598 | lancedb | bge_small_en | VECTOR_L2 | 476.032 | 469.18 | 487.054 | 5.607 | 473.812 | 485.595 |
| 599 | lancedb | gte_small | VECTOR_limit20 | 476.063 | 469.631 | 495.699 | 7.152 | 474.018 | 488.629 |
| 600 | lancedb | gte_base | VECTOR_limit5 | 476.169 | 467.512 | 531.087 | 18.445 | 470.079 | 505.862 |
| 601 | lancedb | mpnet_multi_base | VECTOR_default | 476.374 | 473.217 | 481.819 | 2.453 | 475.592 | 480.832 |
| 602 | lancedb | gte_base | VECTOR_cosine | 476.406 | 473.342 | 477.87 | 1.188 | 476.51 | 477.728 |
| 603 | lancedb | e5_base | VECTOR_limit5 | 476.955 | 469.712 | 523.335 | 15.529 | 471.819 | 501.182 |
| 604 | lancedb | mpnet_multi | VECTOR_default | 477.063 | 472.944 | 485.322 | 4.269 | 474.976 | 484.807 |
| 605 | lancedb | mpnet_multi_base | VECTOR_limit5 | 477.175 | 467.304 | 520.413 | 14.53 | 473.033 | 499.529 |
| 606 | lancedb | bge_small_en | VECTOR_limit20 | 477.414 | 471.576 | 483.078 | 3.475 | 476.902 | 482.559 |
| 607 | lancedb | all_mini_l6 | VECTOR_limit20 | 477.781 | 472.999 | 488.626 | 4.133 | 476.799 | 484.575 |
| 608 | lancedb | mpnet_multi_base | VECTOR_cosine | 478.136 | 475.872 | 484.155 | 2.239 | 477.672 | 481.787 |
| 609 | lancedb | gte_base | VECTOR_L2 | 478.343 | 473.317 | 487.831 | 4.125 | 477.525 | 485.184 |
| 610 | lancedb | mpnet_multi | VECTOR_limit5 | 478.653 | 471.129 | 524.587 | 15.433 | 473.298 | 503.169 |
| 611 | lancedb | mpnet_base | VECTOR_default | 478.941 | 476.815 | 490.479 | 3.946 | 477.337 | 485.342 |
| 612 | lancedb | distilroberta | VECTOR_L2 | 479.217 | 475.666 | 482.809 | 2.204 | 479.18 | 482.229 |
| 613 | lancedb | minilm_l12 | VECTOR_limit20 | 479.326 | 472.155 | 487.972 | 4.176 | 479.383 | 486.037 |
| 614 | lancedb | mpnet_multi_base | VECTOR_L2 | 479.363 | 477.588 | 482.201 | 1.49 | 478.875 | 481.916 |
| 615 | lancedb | mpnet_base | VECTOR_limit5 | 479.465 | 468.056 | 529.338 | 16.946 | 474.222 | 507.118 |
| 616 | lancedb | e5_small_base | VECTOR_limit20 | 479.61 | 474.721 | 482.781 | 2.683 | 481.01 | 482.188 |
| 617 | lancedb | e5_small_hf | VECTOR_limit20 | 479.833 | 472.348 | 485.909 | 4.459 | 479.946 | 485.45 |
| 618 | lancedb | mpnet_multi | VECTOR_cosine | 480.064 | 475.411 | 484.794 | 3.101 | 480.091 | 484.739 |
| 619 | lancedb | mpnet_base | VECTOR_cosine | 480.241 | 473.852 | 486.062 | 3.664 | 480.443 | 485.514 |
| 620 | lancedb | e5_base_hf | VECTOR_default | 480.295 | 474.93 | 493.837 | 5.024 | 478.927 | 489.04 |
| 621 | lancedb | distilroberta | VECTOR_cosine | 480.3 | 474.276 | 483.82 | 2.779 | 480.872 | 483.546 |
| 622 | lancedb | mpnet_multi | VECTOR_L2 | 480.322 | 476.099 | 483.714 | 2.297 | 480.32 | 483.485 |
| 623 | lancedb | e5_base_hf | VECTOR_limit5 | 480.404 | 470.749 | 528.378 | 16.191 | 475.752 | 506.855 |
| 624 | lancedb | e5_base_hf | VECTOR_L2 | 480.523 | 476.978 | 485.912 | 2.393 | 480.643 | 484.5 |
| 625 | lancedb | e5_base_hf | VECTOR_cosine | 480.528 | 477.484 | 483.046 | 1.575 | 480.883 | 482.587 |
| 626 | lancedb | distilroberta | VECTOR_limit5 | 480.87 | 471.092 | 527.048 | 15.552 | 475.418 | 505.67 |
| 627 | lancedb | gte_base | VECTOR_limit20 | 480.926 | 477.556 | 484.938 | 2.621 | 481.266 | 484.367 |
| 628 | lancedb | multi_qa_mpnet | VECTOR_L2 | 481.107 | 473.969 | 485.152 | 3.193 | 481.796 | 484.817 |
| 629 | lancedb | multi_qa_mpnet | VECTOR_default | 481.138 | 475.177 | 488.929 | 3.862 | 481.58 | 486.784 |
| 630 | lancedb | distilroberta | VECTOR_default | 481.168 | 475.921 | 492.598 | 4.957 | 479.623 | 490.7 |
| 631 | lancedb | multi_qa_mpnet | VECTOR_limit5 | 481.254 | 470.236 | 523.326 | 14.397 | 476.996 | 504.429 |
| 632 | lancedb | bge_base_en | VECTOR_default | 481.258 | 475.645 | 486.034 | 2.98 | 481.401 | 485.226 |
| 633 | lancedb | multi_qa_mpnet | VECTOR_cosine | 481.295 | 478.03 | 486.86 | 2.781 | 481.023 | 486.193 |
| 634 | lancedb | bge_base_en | VECTOR_limit5 | 481.319 | 470.313 | 526.086 | 15.266 | 477.786 | 505.668 |
| 635 | lancedb | snowflake_arctic_l | VECTOR_limit5 | 481.695 | 473.489 | 526.56 | 15.072 | 477.855 | 505.26 |
| 636 | lancedb | snowflake_arctic_l_base | VECTOR_limit5 | 481.964 | 472.516 | 524.382 | 14.312 | 477.944 | 504.669 |
| 637 | lancedb | e5_base_base | VECTOR_limit5 | 482.094 | 474.972 | 526.392 | 14.916 | 476.58 | 506.208 |
| 638 | lancedb | e5_base_base | VECTOR_default | 482.172 | 478.101 | 490.85 | 4.014 | 480.476 | 489.453 |
| 639 | lancedb | e5_base_base | VECTOR_L2 | 482.642 | 478.02 | 487.053 | 2.934 | 482.67 | 486.72 |
| 640 | lancedb | bge_base_en | VECTOR_L2 | 482.681 | 478.01 | 485.898 | 2.183 | 483.026 | 485.576 |
| 641 | lancedb | bge_base_en | VECTOR_cosine | 482.695 | 479.008 | 486.106 | 1.878 | 483.038 | 485.179 |
| 642 | lancedb | e5_base_base | VECTOR_cosine | 482.98 | 480.311 | 487.7 | 2.629 | 481.743 | 487.265 |
| 643 | lancedb | snowflake_arctic_l | VECTOR_default | 483.384 | 475.718 | 507.303 | 8.64 | 480.454 | 498.649 |
| 644 | lancedb | mpnet_base | VECTOR_L2 | 483.478 | 476.856 | 504.26 | 7.522 | 481.526 | 496.296 |
| 645 | lancedb | snowflake_arctic_l | VECTOR_cosine | 483.548 | 481.367 | 488.318 | 1.908 | 482.91 | 486.678 |
| 646 | lancedb | bge_squad | VECTOR_default | 483.725 | 479.654 | 491.438 | 3.267 | 482.993 | 489.511 |
| 647 | lancedb | snowflake_arctic_l_base | VECTOR_default | 484.026 | 479.497 | 492.131 | 3.875 | 484.212 | 490.806 |
| 648 | lancedb | e5_small | VECTOR_limit50 | 484.48 | 476.815 | 491.569 | 4.204 | 484.724 | 490.879 |
| 649 | lancedb | snowflake_arctic_l | VECTOR_L2 | 484.617 | 480.03 | 496.608 | 4.413 | 483.734 | 491.891 |
| 650 | lancedb | mpnet_multi | VECTOR_limit20 | 484.933 | 480.985 | 487.188 | 1.741 | 485.42 | 486.721 |
| 651 | lancedb | distilroberta | VECTOR_limit20 | 485.476 | 481.622 | 487.475 | 1.938 | 486.234 | 487.443 |
| 652 | lancedb | snowflake_arctic_l_base | VECTOR_L2 | 485.544 | 483.174 | 490.001 | 1.697 | 485.177 | 488.378 |
| 653 | lancedb | bge_squad | VECTOR_L2 | 485.984 | 481.43 | 488.919 | 2.117 | 486.391 | 488.76 |
| 654 | lancedb | bge_squad | VECTOR_limit5 | 486.5 | 477.793 | 530.95 | 15.084 | 482.368 | 511.197 |
| 655 | lancedb | snowflake_arctic_l_base | VECTOR_cosine | 486.559 | 483.273 | 499.049 | 4.373 | 485.494 | 493.747 |
| 656 | lancedb | bge_squad | VECTOR_cosine | 486.746 | 483.632 | 490.289 | 1.98 | 486.142 | 490.111 |
| 657 | lancedb | all_mini_l6_base | VECTOR_limit50 | 487.054 | 483.523 | 489.867 | 2.168 | 487.651 | 489.573 |
| 658 | lancedb | mpnet_base | VECTOR_limit20 | 487.102 | 484.738 | 489.969 | 1.714 | 486.573 | 489.843 |
| 659 | lancedb | e5_base | VECTOR_limit20 | 487.47 | 479.788 | 508.452 | 9.752 | 483.632 | 506.7 |
| 660 | lancedb | mpnet_multi_base | VECTOR_limit20 | 487.632 | 482.621 | 512.489 | 8.364 | 485.254 | 500.842 |
| 661 | lancedb | gte_small | VECTOR_limit50 | 487.882 | 484.619 | 493.063 | 2.656 | 486.873 | 492.354 |
| 662 | lancedb | e5_base_hf | VECTOR_limit20 | 487.956 | 485.802 | 490.419 | 1.644 | 487.783 | 490.368 |
| 663 | lancedb | bge_base_en | VECTOR_limit20 | 488.118 | 484.587 | 492.506 | 2.099 | 488.164 | 491.321 |
| 664 | lancedb | multi_qa_mpnet | VECTOR_limit20 | 488.16 | 483.693 | 491.125 | 2.135 | 488.149 | 490.798 |
| 665 | lancedb | multi_qa_minilm | VECTOR_limit50 | 488.606 | 485.297 | 494.063 | 3.12 | 487.311 | 493.759 |
| 666 | lancedb | bge_squad_base | VECTOR_limit5 | 488.955 | 480.428 | 535.688 | 15.672 | 484.045 | 513.731 |
| 667 | lancedb | bge_squad_base | VECTOR_default | 489.064 | 485.019 | 495.562 | 3.345 | 487.771 | 494.12 |
| 668 | lancedb | e5_base_base | VECTOR_limit20 | 489.704 | 487.177 | 493.505 | 2.159 | 489.57 | 492.809 |
| 669 | lancedb | bge_squad_base | VECTOR_L2 | 489.716 | 484.441 | 498.936 | 4.268 | 489.033 | 497.4 |
| 670 | lancedb | snowflake_arctic_l | VECTOR_limit20 | 489.979 | 484.925 | 492.279 | 1.957 | 490.212 | 492.23 |
| 671 | lancedb | bge_squad_base | VECTOR_cosine | 490.528 | 487.794 | 496.8 | 2.422 | 490.461 | 494.459 |
| 672 | lancedb | snowflake_arctic_l_base | VECTOR_limit20 | 490.625 | 487.41 | 494.428 | 2.208 | 490.667 | 493.721 |
| 673 | lancedb | e5_base | VECTOR_cosine | 491.738 | 476.54 | 532.195 | 16.262 | 486.847 | 519.953 |
| 674 | lancedb | minilm_l12 | VECTOR_limit50 | 491.921 | 486.544 | 499.494 | 3.299 | 491.656 | 496.956 |
| 675 | lancedb | bge_squad | VECTOR_limit20 | 492 | 489.313 | 496.698 | 2.368 | 491.065 | 495.785 |
| 676 | lancedb | all_mini_l6 | VECTOR_limit50 | 492.323 | 490.149 | 497.041 | 2.015 | 491.769 | 495.891 |
| 677 | lancedb | bge_small_en | VECTOR_limit50 | 492.861 | 486.229 | 500.659 | 4.062 | 492.06 | 499.626 |
| 678 | lancedb | paraphrase_multi | VECTOR_limit50 | 493.425 | 486.087 | 500.485 | 4.078 | 493.1 | 499.265 |
| 679 | lancedb | e5_small_hf | VECTOR_limit50 | 494.406 | 490.087 | 502.24 | 3.873 | 493.815 | 501.014 |
| 680 | lancedb | bge_squad_base | VECTOR_limit20 | 495.585 | 492.213 | 500.216 | 2.603 | 495.213 | 500.044 |
| 681 | lancedb | gte_base | VECTOR_limit50 | 496.278 | 494.479 | 498.01 | 1.113 | 496.025 | 497.938 |
| 682 | lancedb | e5_small_base | VECTOR_limit50 | 496.571 | 485.705 | 503.707 | 5.117 | 496.372 | 502.76 |
| 683 | lancedb | distilroberta | VECTOR_limit50 | 499.849 | 496.408 | 502.275 | 2.027 | 500.029 | 502.197 |
| 684 | lancedb | mpnet_base | VECTOR_limit50 | 500.702 | 496.755 | 503.942 | 2.321 | 500.183 | 503.789 |
| 685 | lancedb | mpnet_multi | VECTOR_limit50 | 502.195 | 499.193 | 507.119 | 2.386 | 501.943 | 506.093 |
| 686 | lancedb | mpnet_multi_base | VECTOR_limit50 | 502.858 | 499.577 | 512.911 | 3.905 | 501.724 | 509.695 |
| 687 | lancedb | bge_base_en | VECTOR_limit50 | 505.104 | 501.124 | 509.489 | 2.769 | 505.304 | 509.303 |
| 688 | lancedb | snowflake_arctic_l | VECTOR_limit50 | 505.943 | 503.245 | 508.737 | 1.499 | 505.512 | 508.296 |
| 689 | lancedb | e5_base_hf | VECTOR_limit50 | 506.288 | 500.03 | 535.889 | 9.982 | 503.463 | 522.052 |
| 690 | lancedb | multi_qa_mpnet | VECTOR_limit50 | 507.223 | 503.861 | 512.445 | 2.408 | 506.924 | 511.044 |
| 691 | lancedb | bge_squad | VECTOR_limit50 | 507.507 | 505.27 | 510.311 | 1.688 | 507.683 | 510.1 |
| 692 | lancedb | e5_base_base | VECTOR_limit50 | 507.833 | 503.508 | 514.626 | 3.332 | 506.572 | 513.793 |
| 693 | lancedb | snowflake_arctic_l_base | VECTOR_limit50 | 511.25 | 505.803 | 536.662 | 8.841 | 507.903 | 526.351 |
| 694 | lancedb | e5_base | VECTOR_L2 | 512.725 | 468.917 | 780.022 | 91.815 | 476.241 | 675.952 |
| 695 | lancedb | bge_squad_base | VECTOR_limit50 | 514.504 | 511.243 | 521.983 | 3.027 | 513.41 | 519.605 |
| 696 | lancedb | e5_base | VECTOR_limit50 | 526.519 | 520.481 | 530.596 | 3.309 | 527.247 | 530.506 |

## MILVUS

_MILVUS — Model x Algoritma (ms ortalama)_  
| Model | HNSW_batch | HNSW_default | HNSW_limit20 | HNSW_limit5 | HNSW_limit50 |
|---|---|---|---|---|---|
| e5_small | 9.734 | 80.814 | 79.278 | 78.843 | 80.596 |
| mpnet_multi | 17.696 | 131.752 | 133.339 | 130.268 | 131.685 |
| e5_base | 16.841 | 129.113 | 128.783 | 128.59 | 130.473 |
| bge_squad | 23.598 | 166.793 | 168.097 | 167.763 | 167.385 |
| qwen_lora | - | - | - | - | - |
| snowflake_arctic_l | 23.109 | 173.956 | 170.418 | 170.208 | 171.06 |
| all_mini_l6 | 10.043 | 78.274 | 78.615 | 77.302 | 78.927 |
| bge-m3-fine | - | - | - | - | - |
| e5_small_base | 10.019 | 77.734 | 83.749 | 78.093 | 81.179 |
| mpnet_multi_base | 17.062 | 128.168 | 129.123 | 128.227 | 129.365 |
| e5_base_base | 16.451 | 129.279 | 130.784 | 129.156 | 132.39 |
| bge_squad_base | 22.244 | 160.2 | 160.19 | 159.339 | 160.91 |
| qwen_lora_base | - | - | - | - | - |
| snowflake_arctic_l_base | 21.906 | 162.523 | 162.093 | 161.148 | 164.768 |
| all_mini_l6_base | 9.945 | 78.183 | 77.704 | 79.883 | 79.673 |
| bge_m3_base | - | - | - | - | - |
| minilm_l12 | 10.346 | 79.236 | 78.091 | 79.176 | 80.743 |
| mpnet_base | 17.591 | 131.031 | 134.216 | 131.981 | 132.601 |
| distilroberta | 17.811 | 131.517 | 130.862 | 131.764 | 132.383 |
| multi_qa_minilm | 10.302 | 79.338 | 78.782 | 79.334 | 80.211 |
| multi_qa_mpnet | 16.609 | 130.164 | 129.832 | 129.715 | 131.178 |
| paraphrase_multi | 9.824 | 77.33 | 76.441 | 77.187 | 77.87 |
| bge_small_en | 10.292 | 79.196 | 75.179 | 77.276 | 76.886 |
| bge_base_en | 16.8 | 129.182 | 126.964 | 127.629 | 128.469 |
| gte_small | 10.15 | 77.519 | 78.116 | 79.105 | 78.772 |
| gte_base | 17.839 | 129.401 | 129.624 | 129.171 | 131.728 |
| e5_small_hf | 9.973 | 77.689 | 78.462 | 78.445 | 79.833 |
| e5_base_hf | 17.386 | 131.398 | 132.236 | 131.252 | 132.462 |

## QDRANT

_QDRANT — Model x Algoritma (ms ortalama)_  
| Model | EXACT | HNSW_default | HNSW_ef128 | HNSW_ef256 | HNSW_ef32 |
|---|---|---|---|---|---|
| e5_small | 32.728 | 33.634 | 31.982 | 32.675 | 31.472 |
| mpnet_multi | 56.456 | 24.96 | 26.142 | 31.713 | 19.555 |
| e5_base | 55.686 | 29.695 | 31.869 | 40.408 | 24.096 |
| bge_squad | 73.479 | 27.831 | 29.876 | 36.507 | 22.791 |
| qwen_lora | - | - | - | - | - |
| snowflake_arctic_l | 74.538 | 29.649 | 32.302 | 41.072 | 23.441 |
| all_mini_l6 | 32.291 | 31.183 | 32.785 | 33.112 | 32.249 |
| bge-m3-fine | - | - | - | - | - |
| e5_small_base | 32.478 | 33.974 | 32.789 | 31.407 | 32.597 |
| mpnet_multi_base | 56.143 | 26.513 | 27.466 | 32.355 | 22.545 |
| e5_base_base | 57.747 | 25.612 | 26.913 | 32.057 | 20.864 |
| bge_squad_base | 73.632 | 27.009 | 28.563 | 34.757 | 21.73 |
| qwen_lora_base | - | - | - | - | - |
| snowflake_arctic_l_base | 69.713 | 27.681 | 28.886 | 35.355 | 22.332 |
| all_mini_l6_base | 31.891 | 32.597 | 32.494 | 32.274 | 32.369 |
| bge_m3_base | - | - | - | - | - |
| minilm_l12 | 32.396 | 32.357 | 31.262 | 31.329 | 31.522 |
| mpnet_base | 57.963 | 24.734 | 25.044 | 28.84 | 20.528 |
| distilroberta | 57.878 | 24.96 | 25.997 | 30.3 | 21.631 |
| multi_qa_minilm | 32.346 | 31.968 | 32.434 | 33.123 | 32.614 |
| multi_qa_mpnet | 57.99 | 26.282 | 27.778 | 33.254 | 23.034 |
| paraphrase_multi | 31.356 | 31.698 | 30.412 | 30.436 | 30.885 |
| bge_small_en | 32.152 | 31.744 | 31.662 | 31.12 | 31.906 |
| bge_base_en | 57.943 | 25.956 | 26.523 | 31.075 | 21.606 |
| gte_small | 31.688 | 31.875 | 31.79 | 32.468 | 31.795 |
| gte_base | 58.965 | 25.468 | 25.81 | 32.549 | 20.389 |
| e5_small_hf | 32.552 | 32.402 | 32.537 | 32.958 | 31.691 |
| e5_base_hf | 56.352 | 49.591 | 28.778 | 34.041 | 22.456 |

## CHROMADB

_CHROMADB — Model x Algoritma (ms ortalama)_  
| Model | HNSW_batch | HNSW_default | HNSW_n20 | HNSW_n5 | HNSW_n50 |
|---|---|---|---|---|---|
| e5_small | 4.355 | 8.503 | 9.587 | 7.831 | 13.433 |
| mpnet_multi | 5.822 | 10.036 | 11.32 | 9.272 | 15.307 |
| e5_base | 7.029 | 11.419 | 12.423 | 10.154 | 16.401 |
| bge_squad | 7.511 | 10.906 | 11.963 | 11.068 | 15.3 |
| qwen_lora | - | - | - | - | - |
| snowflake_arctic_l | 8.038 | 12.222 | 13.1 | 11.264 | 17.253 |
| all_mini_l6 | 3.808 | 8.597 | 9.237 | 7.887 | 11.935 |
| bge-m3-fine | - | - | - | - | - |
| e5_small_base | 3.571 | 7.766 | 9.453 | 7.121 | 12.146 |
| mpnet_multi_base | 5.337 | 9.678 | 11.082 | 8.975 | 14.797 |
| e5_base_base | 6.182 | 9.784 | 11.202 | 9.136 | 15.551 |
| bge_squad_base | 7.104 | 11.108 | 12.189 | 9.987 | 16.41 |
| qwen_lora_base | - | - | - | - | - |
| snowflake_arctic_l_base | 6.829 | 11.191 | 13.511 | 10.802 | 16.081 |
| all_mini_l6_base | 3.187 | 7.65 | 9.136 | 6.962 | 11.713 |
| bge_m3_base | - | - | - | - | - |
| minilm_l12 | 2.955 | 7.944 | 8.143 | 6.397 | 11.552 |
| mpnet_base | 4.528 | 9.078 | 10.347 | 8.214 | 14.198 |
| distilroberta | 4.665 | 8.947 | 10.332 | 8.24 | 14.196 |
| multi_qa_minilm | 3.501 | 7.571 | 9.042 | 7.371 | 13.191 |
| multi_qa_mpnet | 5.711 | 10.304 | 11.934 | 9.45 | 15.487 |
| paraphrase_multi | 3.527 | 7.614 | 9.371 | 6.798 | 13.256 |
| bge_small_en | 3.492 | 7.973 | 8.9 | 7.574 | 12.014 |
| bge_base_en | 5.206 | 9.1 | 10.671 | 8.614 | 14.362 |
| gte_small | 3.268 | 7.629 | 8.697 | 6.821 | 11.762 |
| gte_base | 5.207 | 9.326 | 10.573 | 8.658 | 14.42 |
| e5_small_hf | 3.773 | 8.162 | 9.823 | 7.633 | 13.005 |
| e5_base_hf | 5.783 | 10.171 | 11.251 | 9.43 | 15.35 |

## LANCEDB

_LANCEDB — Model x Algoritma (ms ortalama)_  
| Model | VECTOR_L2 | VECTOR_cosine | VECTOR_default | VECTOR_limit20 | VECTOR_limit5 | VECTOR_limit50 |
|---|---|---|---|---|---|---|
| e5_small | 469.833 | 460.358 | 451.783 | 467.504 | 458.725 | 484.48 |
| mpnet_multi | 480.322 | 480.064 | 477.063 | 484.933 | 478.653 | 502.195 |
| e5_base | 512.725 | 491.738 | 474.639 | 487.47 | 476.955 | 526.519 |
| bge_squad | 485.984 | 486.746 | 483.725 | 492 | 486.5 | 507.507 |
| qwen_lora | - | - | - | - | - | - |
| snowflake_arctic_l | 484.617 | 483.548 | 483.384 | 489.979 | 481.695 | 505.943 |
| all_mini_l6 | 474.04 | 472.735 | 468.509 | 477.781 | 474.256 | 492.323 |
| bge-m3-fine | - | - | - | - | - | - |
| e5_small_base | 473.446 | 473.205 | 472.292 | 479.61 | 470.086 | 496.571 |
| mpnet_multi_base | 479.363 | 478.136 | 476.374 | 487.632 | 477.175 | 502.858 |
| e5_base_base | 482.642 | 482.98 | 482.172 | 489.704 | 482.094 | 507.833 |
| bge_squad_base | 489.716 | 490.528 | 489.064 | 495.585 | 488.955 | 514.504 |
| qwen_lora_base | - | - | - | - | - | - |
| snowflake_arctic_l_base | 485.544 | 486.559 | 484.026 | 490.625 | 481.964 | 511.25 |
| all_mini_l6_base | 465.848 | 465.835 | 464.431 | 472.198 | 464.508 | 487.054 |
| bge_m3_base | - | - | - | - | - | - |
| minilm_l12 | 473.026 | 473.038 | 474.029 | 479.326 | 474.802 | 491.921 |
| mpnet_base | 483.478 | 480.241 | 478.941 | 487.102 | 479.465 | 500.702 |
| distilroberta | 479.217 | 480.3 | 481.168 | 485.476 | 480.87 | 499.849 |
| multi_qa_minilm | 467.999 | 464.111 | 461.814 | 474.162 | 465.176 | 488.606 |
| multi_qa_mpnet | 481.107 | 481.295 | 481.138 | 488.16 | 481.254 | 507.223 |
| paraphrase_multi | 473.91 | 470.295 | 471.66 | 475.51 | 469.867 | 493.425 |
| bge_small_en | 476.032 | 472.707 | 472.593 | 477.414 | 471.662 | 492.861 |
| bge_base_en | 482.681 | 482.695 | 481.258 | 488.118 | 481.319 | 505.104 |
| gte_small | 467.224 | 466.035 | 467.021 | 476.063 | 468.712 | 487.882 |
| gte_base | 478.343 | 476.406 | 474.25 | 480.926 | 476.169 | 496.278 |
| e5_small_hf | 473.68 | 469.577 | 469.366 | 479.833 | 473.292 | 494.406 |
| e5_base_hf | 480.523 | 480.528 | 480.295 | 487.956 | 480.404 | 506.288 |

## WEAVIATE

_WEAVIATE — Model x Algoritma (ms ortalama)_  
| Model | BM25 | HNSW_default | HNSW_limit20 | HNSW_limit5 | HNSW_limit50 | HYBRID_alpha0.25 | HYBRID_alpha0.5 | HYBRID_alpha0.75 |
|---|---|---|---|---|---|---|---|---|
| e5_small | 10.012 | 14.046 | 18.982 | 11.628 | 37.064 | 41.71 | 50.823 | 59.726 |
| mpnet_multi | 11.014 | 12.993 | 20.364 | 12.324 | 35.248 | 40.252 | 39.983 | 40.167 |
| e5_base | 10.575 | 15.551 | 22.62 | 14.402 | 44.04 | 49.605 | 50.314 | 49.934 |
| bge_squad | 10.287 | 13.039 | 19.48 | 12.155 | 37.128 | 42.267 | 42.629 | 42.6 |
| qwen_lora | - | - | - | - | - | - | - | - |
| snowflake_arctic_l | 10.435 | 14.939 | 21.352 | 13.166 | 39.363 | 46.394 | 46.375 | 46.721 |
| all_mini_l6 | 10.839 | 11.93 | 18.16 | 10.828 | 35.507 | 40.132 | 43.739 | 40.029 |
| bge-m3-fine | - | - | - | - | - | - | - | - |
| e5_small_base | 10.271 | 11.073 | 16.733 | 9.424 | 31.318 | 34.543 | 33.719 | 34.742 |
| mpnet_multi_base | 10.694 | 12.414 | 16.54 | 10.4 | 32.149 | 40.892 | 35.626 | 33.443 |
| e5_base_base | 10.221 | 12.488 | 18.721 | 11.386 | 35.917 | 41.47 | 40.882 | 39.047 |
| bge_squad_base | 10.105 | 13.059 | 18.944 | 11.901 | 33.473 | 38.87 | 37.673 | 38.76 |
| qwen_lora_base | - | - | - | - | - | - | - | - |
| snowflake_arctic_l_base | 10.016 | 12.839 | 19.012 | 11.336 | 34.82 | 39.938 | 39.412 | 38.924 |
| all_mini_l6_base | 10.329 | 9.147 | 12.795 | 8.062 | 25.32 | 27.518 | 26.666 | 26.751 |
| bge_m3_base | - | - | - | - | - | - | - | - |
| minilm_l12 | 10.15 | 8.872 | 13.599 | 7.979 | 26.127 | 27.917 | 26.57 | 26.7 |
| mpnet_base | 10.329 | 9.927 | 14.512 | 8.87 | 27.521 | 30.043 | 28.704 | 28.818 |
| distilroberta | 9.964 | 10.109 | 14.179 | 9.45 | 28.244 | 32.036 | 30.195 | 30.121 |
| multi_qa_minilm | 9.986 | 14.763 | 15.92 | 9.793 | 29.803 | 32.409 | 31.056 | 30.919 |
| multi_qa_mpnet | 9.993 | 12.164 | 17.548 | 10.782 | 33.308 | 37.118 | 36.421 | 36.22 |
| paraphrase_multi | 10.038 | 9.761 | 13.745 | 8.143 | 27.546 | 29.177 | 28.57 | 28.595 |
| bge_small_en | 9.779 | 10.06 | 13.722 | 8.61 | 28.243 | 30.666 | 29.476 | 29.867 |
| bge_base_en | 9.994 | 11.092 | 16.151 | 9.697 | 29.258 | 31.752 | 31.793 | 33.264 |
| gte_small | 9.958 | 9.912 | 15.672 | 8.708 | 28.439 | 31.625 | 30.571 | 31.666 |
| gte_base | 9.933 | 11.447 | 17.009 | 10.188 | 32.736 | 34.868 | 35.559 | 34.585 |
| e5_small_hf | 9.929 | 10.83 | 16.556 | 9.461 | 31.311 | 34.409 | 38.625 | 36.433 |
| e5_base_hf | 9.984 | 12.461 | 19.067 | 11.362 | 35.891 | 40.288 | 40.962 | 40.995 |

## Bulma Basarimi

_BELGE BULMA BASARIMI — SQuAD (akademik / BEIR-style)_  
_Sorgu ornek: 1000  top_k: 100  metrik: nDCG@10 (birincil), Recall@10/100, MRR@10, Hit@1/10_  
| Sira | Veritabani | Model | ndcg@10 | recall@10 | recall@100 | mrr@10 | hit@1 | hit@10 | search_time_sec |
|---|---|---|---|---|---|---|---|---|---|
| 1 | milvus | bge_squad | 0.9008 | 0.97 | 1 | 0.8778 | 0.814 | 0.97 | 3.585 |
| 2 | qdrant | bge_squad | 0.9008 | 0.97 | 1 | 0.8778 | 0.814 | 0.97 | 6.356 |
| 3 | chromadb | bge_squad | 0.9008 | 0.97 | 1 | 0.8778 | 0.814 | 0.97 | 3.786 |
| 4 | lancedb | bge_squad | 0.9008 | 0.97 | 1 | 0.8778 | 0.814 | 0.97 | 7.173 |
| 5 | weaviate | bge_squad | 0.9008 | 0.97 | 1 | 0.8778 | 0.814 | 0.97 | 3.762 |
| 6 | milvus | snowflake_arctic_l | 0.827 | 0.914 | 0.985 | 0.7988 | 0.734 | 0.914 | 3.845 |
| 7 | qdrant | snowflake_arctic_l | 0.827 | 0.914 | 0.985 | 0.7988 | 0.734 | 0.914 | 6.173 |
| 8 | chromadb | snowflake_arctic_l | 0.827 | 0.914 | 0.985 | 0.7988 | 0.734 | 0.914 | 3.99 |
| 9 | weaviate | snowflake_arctic_l | 0.827 | 0.914 | 0.985 | 0.7988 | 0.734 | 0.914 | 4.998 |
| 10 | lancedb | snowflake_arctic_l | 0.8269 | 0.915 | 0.985 | 0.7983 | 0.733 | 0.915 | 9.142 |
| 11 | milvus | all_mini_l6 | 0.8188 | 0.928 | 0.989 | 0.7831 | 0.697 | 0.928 | 3.367 |
| 12 | qdrant | all_mini_l6 | 0.8188 | 0.928 | 0.989 | 0.7831 | 0.697 | 0.928 | 4.863 |
| 13 | chromadb | all_mini_l6 | 0.8188 | 0.928 | 0.989 | 0.7831 | 0.697 | 0.928 | 3.33 |
| 14 | weaviate | all_mini_l6 | 0.8188 | 0.928 | 0.989 | 0.7831 | 0.697 | 0.928 | 5.018 |
| 15 | lancedb | all_mini_l6 | 0.8181 | 0.925 | 0.989 | 0.7832 | 0.698 | 0.925 | 6.984 |
| 16 | milvus | e5_small_hf | 0.5437 | 0.671 | 0.857 | 0.5036 | 0.427 | 0.671 | 4.201 |
| 17 | qdrant | e5_small_hf | 0.5437 | 0.671 | 0.857 | 0.5036 | 0.427 | 0.671 | 6.534 |
| 18 | lancedb | e5_small_hf | 0.5437 | 0.671 | 0.857 | 0.5036 | 0.427 | 0.671 | 9.592 |
| 19 | weaviate | e5_small_hf | 0.5431 | 0.67 | 0.856 | 0.5031 | 0.427 | 0.67 | 4.91 |
| 20 | chromadb | e5_small_hf | 0.5324 | 0.657 | 0.838 | 0.493 | 0.418 | 0.657 | 4.264 |
| 21 | milvus | e5_base_hf | 0.526 | 0.645 | 0.852 | 0.4885 | 0.416 | 0.645 | 4.49 |
| 22 | qdrant | e5_base_hf | 0.526 | 0.645 | 0.852 | 0.4885 | 0.416 | 0.645 | 6.965 |
| 23 | lancedb | e5_base_hf | 0.526 | 0.645 | 0.852 | 0.4885 | 0.416 | 0.645 | 11.208 |
| 24 | weaviate | e5_base_hf | 0.526 | 0.645 | 0.852 | 0.4885 | 0.416 | 0.645 | 5.279 |
| 25 | chromadb | e5_base_hf | 0.5212 | 0.639 | 0.834 | 0.484 | 0.413 | 0.639 | 4.103 |
| 26 | milvus | e5_base_base | 0.5118 | 0.628 | 0.821 | 0.4748 | 0.397 | 0.628 | 3.675 |
| 27 | qdrant | e5_base_base | 0.5118 | 0.628 | 0.821 | 0.4748 | 0.397 | 0.628 | 6.359 |
| 28 | lancedb | e5_base_base | 0.5118 | 0.628 | 0.821 | 0.4748 | 0.397 | 0.628 | 7.867 |
| 29 | weaviate | e5_base_base | 0.5118 | 0.628 | 0.821 | 0.4748 | 0.397 | 0.628 | 4.522 |
| 30 | milvus | e5_small_base | 0.4929 | 0.622 | 0.827 | 0.4525 | 0.377 | 0.622 | 3.27 |
| 31 | qdrant | e5_small_base | 0.4929 | 0.622 | 0.827 | 0.4525 | 0.377 | 0.622 | 4.47 |
| 32 | lancedb | e5_small_base | 0.4929 | 0.622 | 0.827 | 0.4525 | 0.377 | 0.622 | 8.074 |
| 33 | weaviate | e5_small_base | 0.4929 | 0.622 | 0.827 | 0.4525 | 0.377 | 0.622 | 4.979 |
| 34 | chromadb | e5_base_base | 0.4917 | 0.603 | 0.786 | 0.4563 | 0.382 | 0.603 | 4.334 |
| 35 | milvus | snowflake_arctic_l_base | 0.4895 | 0.61 | 0.815 | 0.4516 | 0.378 | 0.61 | 3.96 |
| 36 | qdrant | snowflake_arctic_l_base | 0.4895 | 0.61 | 0.815 | 0.4516 | 0.378 | 0.61 | 6.635 |
| 37 | lancedb | snowflake_arctic_l_base | 0.4895 | 0.61 | 0.815 | 0.4516 | 0.378 | 0.61 | 9.113 |
| 38 | weaviate | snowflake_arctic_l_base | 0.4885 | 0.609 | 0.814 | 0.4506 | 0.377 | 0.609 | 4.906 |
| 39 | milvus | bge_squad_base | 0.4876 | 0.611 | 0.828 | 0.4485 | 0.371 | 0.611 | 4.65 |
| 40 | qdrant | bge_squad_base | 0.4876 | 0.611 | 0.828 | 0.4485 | 0.371 | 0.611 | 8.756 |
| 41 | lancedb | bge_squad_base | 0.4876 | 0.611 | 0.828 | 0.4485 | 0.371 | 0.611 | 11.303 |
| 42 | weaviate | bge_squad_base | 0.4876 | 0.611 | 0.828 | 0.4485 | 0.371 | 0.611 | 6.219 |
| 43 | chromadb | snowflake_arctic_l_base | 0.4825 | 0.602 | 0.8 | 0.4451 | 0.373 | 0.602 | 4.126 |
| 44 | chromadb | e5_small_base | 0.4761 | 0.599 | 0.795 | 0.4375 | 0.364 | 0.599 | 3.196 |
| 45 | lancedb | gte_base | 0.4752 | 0.594 | 0.788 | 0.4377 | 0.364 | 0.594 | 9.078 |
| 46 | milvus | gte_base | 0.4748 | 0.594 | 0.788 | 0.4372 | 0.363 | 0.594 | 3.056 |
| 47 | qdrant | gte_base | 0.4748 | 0.594 | 0.788 | 0.4372 | 0.363 | 0.594 | 5.426 |
| 48 | weaviate | gte_base | 0.4748 | 0.594 | 0.789 | 0.4372 | 0.363 | 0.594 | 4.273 |
| 49 | chromadb | bge_squad_base | 0.4733 | 0.596 | 0.801 | 0.4345 | 0.358 | 0.596 | 4.728 |
| 50 | chromadb | gte_base | 0.4703 | 0.588 | 0.774 | 0.4331 | 0.359 | 0.588 | 4.877 |
| 51 | chromadb | e5_base | 0.4695 | 0.593 | 0.812 | 0.4307 | 0.357 | 0.593 | 3.794 |
| 52 | milvus | e5_base | 0.4695 | 0.593 | 0.82 | 0.4307 | 0.357 | 0.593 | 3.793 |
| 53 | qdrant | e5_base | 0.4695 | 0.593 | 0.82 | 0.4307 | 0.357 | 0.593 | 5.319 |
| 54 | lancedb | e5_base | 0.4695 | 0.593 | 0.82 | 0.4307 | 0.357 | 0.593 | 7.135 |
| 55 | weaviate | e5_base | 0.4695 | 0.593 | 0.82 | 0.4307 | 0.357 | 0.593 | 4.148 |
| 56 | milvus | e5_small | 0.469 | 0.6 | 0.832 | 0.4278 | 0.352 | 0.6 | 4.278 |
| 57 | qdrant | e5_small | 0.469 | 0.6 | 0.832 | 0.4278 | 0.352 | 0.6 | 6.124 |
| 58 | lancedb | e5_small | 0.469 | 0.6 | 0.832 | 0.4278 | 0.352 | 0.6 | 8.422 |
| 59 | weaviate | e5_small | 0.469 | 0.6 | 0.832 | 0.4278 | 0.352 | 0.6 | 4.567 |
| 60 | chromadb | e5_small | 0.4681 | 0.599 | 0.832 | 0.427 | 0.351 | 0.599 | 3.893 |
| 61 | milvus | bge_base_en | 0.4677 | 0.595 | 0.812 | 0.4277 | 0.35 | 0.595 | 4.698 |
| 62 | qdrant | bge_base_en | 0.4677 | 0.595 | 0.812 | 0.4277 | 0.35 | 0.595 | 8.226 |
| 63 | lancedb | bge_base_en | 0.4677 | 0.595 | 0.812 | 0.4277 | 0.35 | 0.595 | 11.313 |
| 64 | weaviate | bge_base_en | 0.4677 | 0.595 | 0.812 | 0.4277 | 0.35 | 0.595 | 6.071 |
| 65 | chromadb | bge_base_en | 0.4649 | 0.589 | 0.797 | 0.4259 | 0.35 | 0.589 | 4.781 |
| 66 | milvus | bge_small_en | 0.4606 | 0.587 | 0.8 | 0.4209 | 0.347 | 0.587 | 3.937 |
| 67 | qdrant | bge_small_en | 0.4606 | 0.587 | 0.8 | 0.4209 | 0.347 | 0.587 | 6.242 |
| 68 | lancedb | bge_small_en | 0.4606 | 0.587 | 0.8 | 0.4209 | 0.347 | 0.587 | 9.317 |
| 69 | weaviate | bge_small_en | 0.4606 | 0.587 | 0.8 | 0.4209 | 0.347 | 0.587 | 5.661 |
| 70 | chromadb | bge_small_en | 0.4558 | 0.581 | 0.789 | 0.4164 | 0.343 | 0.581 | 4.179 |
| 71 | milvus | gte_small | 0.449 | 0.564 | 0.764 | 0.4131 | 0.343 | 0.564 | 4.623 |
| 72 | qdrant | gte_small | 0.449 | 0.564 | 0.764 | 0.4131 | 0.343 | 0.564 | 7.768 |
| 73 | weaviate | gte_small | 0.449 | 0.564 | 0.764 | 0.4131 | 0.343 | 0.564 | 5.995 |
| 74 | lancedb | gte_small | 0.4489 | 0.564 | 0.763 | 0.4129 | 0.343 | 0.564 | 11.704 |
| 75 | chromadb | gte_small | 0.4434 | 0.556 | 0.746 | 0.4083 | 0.34 | 0.556 | 5.226 |
| 76 | milvus | multi_qa_minilm | 0.4277 | 0.558 | 0.795 | 0.387 | 0.313 | 0.558 | 3.476 |
| 77 | qdrant | multi_qa_minilm | 0.4277 | 0.558 | 0.795 | 0.387 | 0.313 | 0.558 | 6.832 |
| 78 | lancedb | multi_qa_minilm | 0.4277 | 0.558 | 0.795 | 0.387 | 0.313 | 0.558 | 10.541 |
| 79 | weaviate | multi_qa_minilm | 0.4277 | 0.558 | 0.795 | 0.387 | 0.313 | 0.558 | 4.462 |
| 80 | milvus | all_mini_l6_base | 0.4195 | 0.555 | 0.747 | 0.3772 | 0.3 | 0.555 | 3.793 |
| 81 | qdrant | all_mini_l6_base | 0.4195 | 0.555 | 0.747 | 0.3772 | 0.3 | 0.555 | 5.793 |
| 82 | lancedb | all_mini_l6_base | 0.4195 | 0.555 | 0.747 | 0.3772 | 0.3 | 0.555 | 8.996 |
| 83 | weaviate | all_mini_l6_base | 0.4195 | 0.555 | 0.747 | 0.3772 | 0.3 | 0.555 | 4.265 |
| 84 | chromadb | multi_qa_minilm | 0.4159 | 0.545 | 0.78 | 0.3756 | 0.302 | 0.545 | 5.404 |
| 85 | chromadb | all_mini_l6_base | 0.4159 | 0.548 | 0.729 | 0.3746 | 0.298 | 0.548 | 3.824 |
| 86 | milvus | multi_qa_mpnet | 0.4131 | 0.547 | 0.785 | 0.3712 | 0.296 | 0.547 | 3.566 |
| 87 | qdrant | multi_qa_mpnet | 0.4131 | 0.547 | 0.785 | 0.3712 | 0.296 | 0.547 | 5.922 |
| 88 | lancedb | multi_qa_mpnet | 0.4131 | 0.547 | 0.785 | 0.3712 | 0.296 | 0.547 | 10.14 |
| 89 | weaviate | multi_qa_mpnet | 0.4131 | 0.547 | 0.785 | 0.3712 | 0.296 | 0.547 | 6.078 |
| 90 | chromadb | multi_qa_mpnet | 0.4011 | 0.531 | 0.753 | 0.3605 | 0.288 | 0.531 | 4.145 |
| 91 | milvus | mpnet_multi | 0.3645 | 0.489 | 0.726 | 0.3257 | 0.256 | 0.489 | 3.974 |
| 92 | qdrant | mpnet_multi | 0.3641 | 0.489 | 0.726 | 0.3252 | 0.255 | 0.489 | 6.412 |
| 93 | lancedb | mpnet_multi | 0.3641 | 0.489 | 0.726 | 0.3252 | 0.255 | 0.489 | 9.159 |
| 94 | weaviate | mpnet_multi | 0.3641 | 0.489 | 0.726 | 0.3252 | 0.255 | 0.489 | 5.096 |
| 95 | chromadb | mpnet_multi | 0.3635 | 0.488 | 0.721 | 0.3247 | 0.255 | 0.488 | 4.646 |
| 96 | milvus | distilroberta | 0.3622 | 0.494 | 0.727 | 0.3212 | 0.249 | 0.494 | 3.526 |
| 97 | qdrant | distilroberta | 0.3622 | 0.494 | 0.727 | 0.3212 | 0.249 | 0.494 | 5.964 |
| 98 | lancedb | distilroberta | 0.3622 | 0.494 | 0.727 | 0.3212 | 0.249 | 0.494 | 8.981 |
| 99 | weaviate | distilroberta | 0.3622 | 0.494 | 0.727 | 0.3212 | 0.249 | 0.494 | 4.236 |
| 100 | chromadb | distilroberta | 0.3579 | 0.489 | 0.704 | 0.317 | 0.245 | 0.489 | 4.059 |
| 101 | milvus | mpnet_base | 0.3469 | 0.484 | 0.712 | 0.3039 | 0.226 | 0.484 | 3.723 |
| 102 | qdrant | mpnet_base | 0.3469 | 0.484 | 0.712 | 0.3039 | 0.226 | 0.484 | 5.967 |
| 103 | lancedb | mpnet_base | 0.3469 | 0.484 | 0.712 | 0.3039 | 0.226 | 0.484 | 8.982 |
| 104 | weaviate | mpnet_base | 0.3469 | 0.484 | 0.712 | 0.3039 | 0.226 | 0.484 | 4.434 |
| 105 | milvus | minilm_l12 | 0.3446 | 0.446 | 0.667 | 0.3126 | 0.25 | 0.446 | 3.262 |
| 106 | qdrant | minilm_l12 | 0.3446 | 0.446 | 0.667 | 0.3126 | 0.25 | 0.446 | 5.632 |
| 107 | lancedb | minilm_l12 | 0.3443 | 0.446 | 0.667 | 0.3121 | 0.249 | 0.446 | 8.947 |
| 108 | weaviate | minilm_l12 | 0.3443 | 0.446 | 0.666 | 0.3121 | 0.249 | 0.446 | 4.384 |
| 109 | chromadb | minilm_l12 | 0.3411 | 0.442 | 0.659 | 0.3093 | 0.247 | 0.442 | 3.831 |
| 110 | chromadb | mpnet_base | 0.3409 | 0.475 | 0.695 | 0.2989 | 0.223 | 0.475 | 4.089 |
| 111 | milvus | mpnet_multi_base | 0.3208 | 0.432 | 0.673 | 0.2861 | 0.223 | 0.432 | 3.403 |
| 112 | qdrant | mpnet_multi_base | 0.3205 | 0.432 | 0.673 | 0.2856 | 0.222 | 0.432 | 5.974 |
| 113 | lancedb | mpnet_multi_base | 0.3205 | 0.432 | 0.673 | 0.2856 | 0.222 | 0.432 | 8.768 |
| 114 | weaviate | mpnet_multi_base | 0.3205 | 0.432 | 0.673 | 0.2856 | 0.222 | 0.432 | 4.648 |
| 115 | chromadb | mpnet_multi_base | 0.3152 | 0.425 | 0.662 | 0.2809 | 0.218 | 0.425 | 3.729 |
| 116 | milvus | paraphrase_multi | 0.2923 | 0.397 | 0.638 | 0.2594 | 0.198 | 0.397 | 4.715 |
| 117 | weaviate | paraphrase_multi | 0.2923 | 0.397 | 0.638 | 0.2594 | 0.198 | 0.397 | 4.275 |
| 118 | qdrant | paraphrase_multi | 0.2919 | 0.397 | 0.638 | 0.2589 | 0.197 | 0.397 | 8.007 |
| 119 | lancedb | paraphrase_multi | 0.2919 | 0.397 | 0.638 | 0.2589 | 0.197 | 0.397 | 10.145 |
| 120 | chromadb | paraphrase_multi | 0.2856 | 0.386 | 0.624 | 0.254 | 0.194 | 0.386 | 4.326 |

## BB_MILVUS

_MILVUS — Bulma Basarimi (Model x Metrik)_  
| Model | ndcg@10 | recall@10 | recall@100 | mrr@10 | hit@1 | hit@10 |
|---|---|---|---|---|---|---|
| e5_small | 0.469 | 0.6 | 0.832 | 0.4278 | 0.352 | 0.6 |
| mpnet_multi | 0.3645 | 0.489 | 0.726 | 0.3257 | 0.256 | 0.489 |
| e5_base | 0.4695 | 0.593 | 0.82 | 0.4307 | 0.357 | 0.593 |
| bge_squad | 0.9008 | 0.97 | 1 | 0.8778 | 0.814 | 0.97 |
| qwen_lora | - | - | - | - | - | - |
| snowflake_arctic_l | 0.827 | 0.914 | 0.985 | 0.7988 | 0.734 | 0.914 |
| all_mini_l6 | 0.8188 | 0.928 | 0.989 | 0.7831 | 0.697 | 0.928 |
| bge-m3-fine | - | - | - | - | - | - |
| e5_small_base | 0.4929 | 0.622 | 0.827 | 0.4525 | 0.377 | 0.622 |
| mpnet_multi_base | 0.3208 | 0.432 | 0.673 | 0.2861 | 0.223 | 0.432 |
| e5_base_base | 0.5118 | 0.628 | 0.821 | 0.4748 | 0.397 | 0.628 |
| bge_squad_base | 0.4876 | 0.611 | 0.828 | 0.4485 | 0.371 | 0.611 |
| qwen_lora_base | - | - | - | - | - | - |
| snowflake_arctic_l_base | 0.4895 | 0.61 | 0.815 | 0.4516 | 0.378 | 0.61 |
| all_mini_l6_base | 0.4195 | 0.555 | 0.747 | 0.3772 | 0.3 | 0.555 |
| bge_m3_base | - | - | - | - | - | - |
| minilm_l12 | 0.3446 | 0.446 | 0.667 | 0.3126 | 0.25 | 0.446 |
| mpnet_base | 0.3469 | 0.484 | 0.712 | 0.3039 | 0.226 | 0.484 |
| distilroberta | 0.3622 | 0.494 | 0.727 | 0.3212 | 0.249 | 0.494 |
| multi_qa_minilm | 0.4277 | 0.558 | 0.795 | 0.387 | 0.313 | 0.558 |
| multi_qa_mpnet | 0.4131 | 0.547 | 0.785 | 0.3712 | 0.296 | 0.547 |
| paraphrase_multi | 0.2923 | 0.397 | 0.638 | 0.2594 | 0.198 | 0.397 |
| bge_small_en | 0.4606 | 0.587 | 0.8 | 0.4209 | 0.347 | 0.587 |
| bge_base_en | 0.4677 | 0.595 | 0.812 | 0.4277 | 0.35 | 0.595 |
| gte_small | 0.449 | 0.564 | 0.764 | 0.4131 | 0.343 | 0.564 |
| gte_base | 0.4748 | 0.594 | 0.788 | 0.4372 | 0.363 | 0.594 |
| e5_small_hf | 0.5437 | 0.671 | 0.857 | 0.5036 | 0.427 | 0.671 |
| e5_base_hf | 0.526 | 0.645 | 0.852 | 0.4885 | 0.416 | 0.645 |

## BB_QDRANT

_QDRANT — Bulma Basarimi (Model x Metrik)_  
| Model | ndcg@10 | recall@10 | recall@100 | mrr@10 | hit@1 | hit@10 |
|---|---|---|---|---|---|---|
| e5_small | 0.469 | 0.6 | 0.832 | 0.4278 | 0.352 | 0.6 |
| mpnet_multi | 0.3641 | 0.489 | 0.726 | 0.3252 | 0.255 | 0.489 |
| e5_base | 0.4695 | 0.593 | 0.82 | 0.4307 | 0.357 | 0.593 |
| bge_squad | 0.9008 | 0.97 | 1 | 0.8778 | 0.814 | 0.97 |
| qwen_lora | - | - | - | - | - | - |
| snowflake_arctic_l | 0.827 | 0.914 | 0.985 | 0.7988 | 0.734 | 0.914 |
| all_mini_l6 | 0.8188 | 0.928 | 0.989 | 0.7831 | 0.697 | 0.928 |
| bge-m3-fine | - | - | - | - | - | - |
| e5_small_base | 0.4929 | 0.622 | 0.827 | 0.4525 | 0.377 | 0.622 |
| mpnet_multi_base | 0.3205 | 0.432 | 0.673 | 0.2856 | 0.222 | 0.432 |
| e5_base_base | 0.5118 | 0.628 | 0.821 | 0.4748 | 0.397 | 0.628 |
| bge_squad_base | 0.4876 | 0.611 | 0.828 | 0.4485 | 0.371 | 0.611 |
| qwen_lora_base | - | - | - | - | - | - |
| snowflake_arctic_l_base | 0.4895 | 0.61 | 0.815 | 0.4516 | 0.378 | 0.61 |
| all_mini_l6_base | 0.4195 | 0.555 | 0.747 | 0.3772 | 0.3 | 0.555 |
| bge_m3_base | - | - | - | - | - | - |
| minilm_l12 | 0.3446 | 0.446 | 0.667 | 0.3126 | 0.25 | 0.446 |
| mpnet_base | 0.3469 | 0.484 | 0.712 | 0.3039 | 0.226 | 0.484 |
| distilroberta | 0.3622 | 0.494 | 0.727 | 0.3212 | 0.249 | 0.494 |
| multi_qa_minilm | 0.4277 | 0.558 | 0.795 | 0.387 | 0.313 | 0.558 |
| multi_qa_mpnet | 0.4131 | 0.547 | 0.785 | 0.3712 | 0.296 | 0.547 |
| paraphrase_multi | 0.2919 | 0.397 | 0.638 | 0.2589 | 0.197 | 0.397 |
| bge_small_en | 0.4606 | 0.587 | 0.8 | 0.4209 | 0.347 | 0.587 |
| bge_base_en | 0.4677 | 0.595 | 0.812 | 0.4277 | 0.35 | 0.595 |
| gte_small | 0.449 | 0.564 | 0.764 | 0.4131 | 0.343 | 0.564 |
| gte_base | 0.4748 | 0.594 | 0.788 | 0.4372 | 0.363 | 0.594 |
| e5_small_hf | 0.5437 | 0.671 | 0.857 | 0.5036 | 0.427 | 0.671 |
| e5_base_hf | 0.526 | 0.645 | 0.852 | 0.4885 | 0.416 | 0.645 |

## BB_CHROMADB

_CHROMADB — Bulma Basarimi (Model x Metrik)_  
| Model | ndcg@10 | recall@10 | recall@100 | mrr@10 | hit@1 | hit@10 |
|---|---|---|---|---|---|---|
| e5_small | 0.4681 | 0.599 | 0.832 | 0.427 | 0.351 | 0.599 |
| mpnet_multi | 0.3635 | 0.488 | 0.721 | 0.3247 | 0.255 | 0.488 |
| e5_base | 0.4695 | 0.593 | 0.812 | 0.4307 | 0.357 | 0.593 |
| bge_squad | 0.9008 | 0.97 | 1 | 0.8778 | 0.814 | 0.97 |
| qwen_lora | - | - | - | - | - | - |
| snowflake_arctic_l | 0.827 | 0.914 | 0.985 | 0.7988 | 0.734 | 0.914 |
| all_mini_l6 | 0.8188 | 0.928 | 0.989 | 0.7831 | 0.697 | 0.928 |
| bge-m3-fine | - | - | - | - | - | - |
| e5_small_base | 0.4761 | 0.599 | 0.795 | 0.4375 | 0.364 | 0.599 |
| mpnet_multi_base | 0.3152 | 0.425 | 0.662 | 0.2809 | 0.218 | 0.425 |
| e5_base_base | 0.4917 | 0.603 | 0.786 | 0.4563 | 0.382 | 0.603 |
| bge_squad_base | 0.4733 | 0.596 | 0.801 | 0.4345 | 0.358 | 0.596 |
| qwen_lora_base | - | - | - | - | - | - |
| snowflake_arctic_l_base | 0.4825 | 0.602 | 0.8 | 0.4451 | 0.373 | 0.602 |
| all_mini_l6_base | 0.4159 | 0.548 | 0.729 | 0.3746 | 0.298 | 0.548 |
| bge_m3_base | - | - | - | - | - | - |
| minilm_l12 | 0.3411 | 0.442 | 0.659 | 0.3093 | 0.247 | 0.442 |
| mpnet_base | 0.3409 | 0.475 | 0.695 | 0.2989 | 0.223 | 0.475 |
| distilroberta | 0.3579 | 0.489 | 0.704 | 0.317 | 0.245 | 0.489 |
| multi_qa_minilm | 0.4159 | 0.545 | 0.78 | 0.3756 | 0.302 | 0.545 |
| multi_qa_mpnet | 0.4011 | 0.531 | 0.753 | 0.3605 | 0.288 | 0.531 |
| paraphrase_multi | 0.2856 | 0.386 | 0.624 | 0.254 | 0.194 | 0.386 |
| bge_small_en | 0.4558 | 0.581 | 0.789 | 0.4164 | 0.343 | 0.581 |
| bge_base_en | 0.4649 | 0.589 | 0.797 | 0.4259 | 0.35 | 0.589 |
| gte_small | 0.4434 | 0.556 | 0.746 | 0.4083 | 0.34 | 0.556 |
| gte_base | 0.4703 | 0.588 | 0.774 | 0.4331 | 0.359 | 0.588 |
| e5_small_hf | 0.5324 | 0.657 | 0.838 | 0.493 | 0.418 | 0.657 |
| e5_base_hf | 0.5212 | 0.639 | 0.834 | 0.484 | 0.413 | 0.639 |

## BB_LANCEDB

_LANCEDB — Bulma Basarimi (Model x Metrik)_  
| Model | ndcg@10 | recall@10 | recall@100 | mrr@10 | hit@1 | hit@10 |
|---|---|---|---|---|---|---|
| e5_small | 0.469 | 0.6 | 0.832 | 0.4278 | 0.352 | 0.6 |
| mpnet_multi | 0.3641 | 0.489 | 0.726 | 0.3252 | 0.255 | 0.489 |
| e5_base | 0.4695 | 0.593 | 0.82 | 0.4307 | 0.357 | 0.593 |
| bge_squad | 0.9008 | 0.97 | 1 | 0.8778 | 0.814 | 0.97 |
| qwen_lora | - | - | - | - | - | - |
| snowflake_arctic_l | 0.8269 | 0.915 | 0.985 | 0.7983 | 0.733 | 0.915 |
| all_mini_l6 | 0.8181 | 0.925 | 0.989 | 0.7832 | 0.698 | 0.925 |
| bge-m3-fine | - | - | - | - | - | - |
| e5_small_base | 0.4929 | 0.622 | 0.827 | 0.4525 | 0.377 | 0.622 |
| mpnet_multi_base | 0.3205 | 0.432 | 0.673 | 0.2856 | 0.222 | 0.432 |
| e5_base_base | 0.5118 | 0.628 | 0.821 | 0.4748 | 0.397 | 0.628 |
| bge_squad_base | 0.4876 | 0.611 | 0.828 | 0.4485 | 0.371 | 0.611 |
| qwen_lora_base | - | - | - | - | - | - |
| snowflake_arctic_l_base | 0.4895 | 0.61 | 0.815 | 0.4516 | 0.378 | 0.61 |
| all_mini_l6_base | 0.4195 | 0.555 | 0.747 | 0.3772 | 0.3 | 0.555 |
| bge_m3_base | - | - | - | - | - | - |
| minilm_l12 | 0.3443 | 0.446 | 0.667 | 0.3121 | 0.249 | 0.446 |
| mpnet_base | 0.3469 | 0.484 | 0.712 | 0.3039 | 0.226 | 0.484 |
| distilroberta | 0.3622 | 0.494 | 0.727 | 0.3212 | 0.249 | 0.494 |
| multi_qa_minilm | 0.4277 | 0.558 | 0.795 | 0.387 | 0.313 | 0.558 |
| multi_qa_mpnet | 0.4131 | 0.547 | 0.785 | 0.3712 | 0.296 | 0.547 |
| paraphrase_multi | 0.2919 | 0.397 | 0.638 | 0.2589 | 0.197 | 0.397 |
| bge_small_en | 0.4606 | 0.587 | 0.8 | 0.4209 | 0.347 | 0.587 |
| bge_base_en | 0.4677 | 0.595 | 0.812 | 0.4277 | 0.35 | 0.595 |
| gte_small | 0.4489 | 0.564 | 0.763 | 0.4129 | 0.343 | 0.564 |
| gte_base | 0.4752 | 0.594 | 0.788 | 0.4377 | 0.364 | 0.594 |
| e5_small_hf | 0.5437 | 0.671 | 0.857 | 0.5036 | 0.427 | 0.671 |
| e5_base_hf | 0.526 | 0.645 | 0.852 | 0.4885 | 0.416 | 0.645 |

## BB_WEAVIATE

_WEAVIATE — Bulma Basarimi (Model x Metrik)_  
| Model | ndcg@10 | recall@10 | recall@100 | mrr@10 | hit@1 | hit@10 |
|---|---|---|---|---|---|---|
| e5_small | 0.469 | 0.6 | 0.832 | 0.4278 | 0.352 | 0.6 |
| mpnet_multi | 0.3641 | 0.489 | 0.726 | 0.3252 | 0.255 | 0.489 |
| e5_base | 0.4695 | 0.593 | 0.82 | 0.4307 | 0.357 | 0.593 |
| bge_squad | 0.9008 | 0.97 | 1 | 0.8778 | 0.814 | 0.97 |
| qwen_lora | - | - | - | - | - | - |
| snowflake_arctic_l | 0.827 | 0.914 | 0.985 | 0.7988 | 0.734 | 0.914 |
| all_mini_l6 | 0.8188 | 0.928 | 0.989 | 0.7831 | 0.697 | 0.928 |
| bge-m3-fine | - | - | - | - | - | - |
| e5_small_base | 0.4929 | 0.622 | 0.827 | 0.4525 | 0.377 | 0.622 |
| mpnet_multi_base | 0.3205 | 0.432 | 0.673 | 0.2856 | 0.222 | 0.432 |
| e5_base_base | 0.5118 | 0.628 | 0.821 | 0.4748 | 0.397 | 0.628 |
| bge_squad_base | 0.4876 | 0.611 | 0.828 | 0.4485 | 0.371 | 0.611 |
| qwen_lora_base | - | - | - | - | - | - |
| snowflake_arctic_l_base | 0.4885 | 0.609 | 0.814 | 0.4506 | 0.377 | 0.609 |
| all_mini_l6_base | 0.4195 | 0.555 | 0.747 | 0.3772 | 0.3 | 0.555 |
| bge_m3_base | - | - | - | - | - | - |
| minilm_l12 | 0.3443 | 0.446 | 0.666 | 0.3121 | 0.249 | 0.446 |
| mpnet_base | 0.3469 | 0.484 | 0.712 | 0.3039 | 0.226 | 0.484 |
| distilroberta | 0.3622 | 0.494 | 0.727 | 0.3212 | 0.249 | 0.494 |
| multi_qa_minilm | 0.4277 | 0.558 | 0.795 | 0.387 | 0.313 | 0.558 |
| multi_qa_mpnet | 0.4131 | 0.547 | 0.785 | 0.3712 | 0.296 | 0.547 |
| paraphrase_multi | 0.2923 | 0.397 | 0.638 | 0.2594 | 0.198 | 0.397 |
| bge_small_en | 0.4606 | 0.587 | 0.8 | 0.4209 | 0.347 | 0.587 |
| bge_base_en | 0.4677 | 0.595 | 0.812 | 0.4277 | 0.35 | 0.595 |
| gte_small | 0.449 | 0.564 | 0.764 | 0.4131 | 0.343 | 0.564 |
| gte_base | 0.4748 | 0.594 | 0.789 | 0.4372 | 0.363 | 0.594 |
| e5_small_hf | 0.5431 | 0.67 | 0.856 | 0.5031 | 0.427 | 0.67 |
| e5_base_hf | 0.526 | 0.645 | 0.852 | 0.4885 | 0.416 | 0.645 |

## Write Throughput

_YAZMA HIZI (kayit/saniye) + VRAM Peak (GB)_  
| Model | DB | kayit/sn | sure(s) | kayit | vram_peak_gb |
|---|---|---|---|---|---|
| e5_small | milvus | 5510 | 9.497 | 52331 | 0 |
| e5_small | qdrant | 2930 | 17.86 | 52331 | 0 |
| e5_small | chromadb | 630.2 | 83.043 | 52331 | 0 |
| e5_small | lancedb | 1380 | 37.926 | 52331 | 0 |
| e5_small | weaviate | 2903 | 18.026 | 52331 | 0 |
| mpnet_multi | milvus | 3138 | 16.675 | 52331 | 0 |
| mpnet_multi | qdrant | 1371 | 38.175 | 52331 | 0 |
| mpnet_multi | chromadb | 571.5 | 91.564 | 52331 | 0 |
| mpnet_multi | lancedb | 1313 | 39.845 | 52331 | 0 |
| mpnet_multi | weaviate | 3767 | 13.892 | 52331 | 0 |
| e5_base | milvus | 3075 | 17.019 | 52331 | 0 |
| e5_base | qdrant | 1350 | 38.778 | 52331 | 0 |
| e5_base | chromadb | 557.7 | 93.833 | 52331 | 0 |
| e5_base | lancedb | 1336 | 39.16 | 52331 | 0 |
| e5_base | weaviate | 3132 | 16.709 | 52331 | 0 |
| bge_squad | milvus | 5106 | 10.25 | 52331 | 0 |
| bge_squad | qdrant | 1267 | 41.31 | 52331 | 0 |
| bge_squad | chromadb | 538.8 | 97.134 | 52331 | 0 |
| bge_squad | lancedb | 1310 | 39.963 | 52331 | 0 |
| bge_squad | weaviate | 3147 | 16.629 | 52331 | 0 |
| snowflake_arctic_l | milvus | 3126 | 16.743 | 52331 | 0 |
| snowflake_arctic_l | qdrant | 1123 | 46.602 | 52331 | 0 |
| snowflake_arctic_l | chromadb | 542.1 | 96.533 | 52331 | 0 |
| snowflake_arctic_l | lancedb | 1286 | 40.692 | 52331 | 0 |
| snowflake_arctic_l | weaviate | 2990 | 17.505 | 52331 | 0 |
| all_mini_l6 | milvus | 3106 | 16.849 | 52331 | 0 |
| all_mini_l6 | qdrant | 2471 | 21.181 | 52331 | 0 |
| all_mini_l6 | chromadb | 604.8 | 86.524 | 52331 | 0 |
| all_mini_l6 | lancedb | 1290 | 40.576 | 52331 | 0 |
| all_mini_l6 | weaviate | 4144 | 12.628 | 52331 | 0 |
| e5_small_base | milvus | 3296 | 15.876 | 52331 | 0 |
| e5_small_base | qdrant | 2505 | 20.894 | 52331 | 0 |
| e5_small_base | chromadb | 601.2 | 87.046 | 52331 | 0 |
| e5_small_base | lancedb | 1330 | 39.337 | 52331 | 0 |
| e5_small_base | weaviate | 4614 | 11.342 | 52331 | 0 |
| mpnet_multi_base | milvus | 3069 | 17.05 | 52331 | 0 |
| mpnet_multi_base | qdrant | 1367 | 38.275 | 52331 | 0 |
| mpnet_multi_base | chromadb | 570.9 | 91.665 | 52331 | 0 |
| mpnet_multi_base | lancedb | 1324 | 39.511 | 52331 | 0 |
| mpnet_multi_base | weaviate | 3516 | 14.886 | 52331 | 0 |
| e5_base_base | milvus | 3088 | 16.945 | 52331 | 0 |
| e5_base_base | qdrant | 1339 | 39.095 | 52331 | 0 |
| e5_base_base | chromadb | 575.6 | 90.923 | 52331 | 0 |
| e5_base_base | lancedb | 1317 | 39.748 | 52331 | 0 |
| e5_base_base | weaviate | 3236 | 16.17 | 52331 | 0 |
| bge_squad_base | milvus | 5033 | 10.398 | 52331 | 0 |
| bge_squad_base | qdrant | 1114 | 46.972 | 52331 | 0 |
| bge_squad_base | chromadb | 554.4 | 94.387 | 52331 | 0 |
| bge_squad_base | lancedb | 1306 | 40.078 | 52331 | 0 |
| bge_squad_base | weaviate | 3401 | 15.385 | 52331 | 0 |
| snowflake_arctic_l_base | milvus | 5078 | 10.306 | 52331 | 0 |
| snowflake_arctic_l_base | qdrant | 1302 | 40.205 | 52331 | 0 |
| snowflake_arctic_l_base | chromadb | 599.1 | 87.356 | 52331 | 0 |
| snowflake_arctic_l_base | lancedb | 1310 | 39.948 | 52331 | 0 |
| snowflake_arctic_l_base | weaviate | 3268 | 16.011 | 52331 | 0 |
| all_mini_l6_base | milvus | 3123 | 16.755 | 52331 | 0 |
| all_mini_l6_base | qdrant | 2354 | 22.227 | 52331 | 0 |
| all_mini_l6_base | chromadb | 594.4 | 88.04 | 52331 | 0 |
| all_mini_l6_base | lancedb | 1342 | 38.992 | 52331 | 0 |
| all_mini_l6_base | weaviate | 4755 | 11.006 | 52331 | 0 |
| minilm_l12 | milvus | 3199 | 16.358 | 52331 | 0 |
| minilm_l12 | qdrant | 2394 | 21.859 | 52331 | 0 |
| minilm_l12 | chromadb | 596.2 | 87.768 | 52331 | 0 |
| minilm_l12 | lancedb | 1342 | 38.984 | 52331 | 0 |
| minilm_l12 | weaviate | 4841 | 10.81 | 52331 | 0 |
| mpnet_base | milvus | 3104 | 16.861 | 52331 | 0 |
| mpnet_base | qdrant | 1362 | 38.417 | 52331 | 0 |
| mpnet_base | chromadb | 572.7 | 91.377 | 52331 | 0 |
| mpnet_base | lancedb | 1329 | 39.363 | 52331 | 0 |
| mpnet_base | weaviate | 3863 | 13.547 | 52331 | 0 |
| distilroberta | milvus | 3354 | 15.603 | 52331 | 0 |
| distilroberta | qdrant | 1464 | 35.734 | 52331 | 0 |
| distilroberta | chromadb | 575 | 91.014 | 52331 | 0 |
| distilroberta | lancedb | 1333 | 39.251 | 52331 | 0 |
| distilroberta | weaviate | 3620 | 14.454 | 52331 | 0 |
| multi_qa_minilm | milvus | 3202 | 16.345 | 52331 | 0 |
| multi_qa_minilm | qdrant | 2367 | 22.111 | 52331 | 0 |
| multi_qa_minilm | chromadb | 593.6 | 88.165 | 52331 | 0 |
| multi_qa_minilm | lancedb | 1340 | 39.069 | 52331 | 0 |
| multi_qa_minilm | weaviate | 4677 | 11.189 | 52331 | 0 |
| multi_qa_mpnet | milvus | 3075 | 17.02 | 52331 | 0 |
| multi_qa_mpnet | qdrant | 1380 | 37.929 | 52331 | 0 |
| multi_qa_mpnet | chromadb | 580.6 | 90.128 | 52331 | 0 |
| multi_qa_mpnet | lancedb | 1342 | 39.006 | 52331 | 0 |
| multi_qa_mpnet | weaviate | 3780 | 13.846 | 52331 | 0 |
| paraphrase_multi | milvus | 3108 | 16.835 | 52331 | 0 |
| paraphrase_multi | qdrant | 2388 | 21.919 | 52331 | 0 |
| paraphrase_multi | chromadb | 584.7 | 89.494 | 52331 | 0 |
| paraphrase_multi | lancedb | 1321 | 39.608 | 52331 | 0 |
| paraphrase_multi | weaviate | 4737 | 11.048 | 52331 | 0 |
| bge_small_en | milvus | 2992 | 17.492 | 52331 | 0 |
| bge_small_en | qdrant | 2252 | 23.243 | 52331 | 0 |
| bge_small_en | chromadb | 599.3 | 87.318 | 52331 | 0 |
| bge_small_en | lancedb | 1328 | 39.415 | 52331 | 0 |
| bge_small_en | weaviate | 4241 | 12.339 | 52331 | 0 |
| bge_base_en | milvus | 3183 | 16.441 | 52331 | 0 |
| bge_base_en | qdrant | 1370 | 38.192 | 52331 | 0 |
| bge_base_en | chromadb | 587 | 89.149 | 52331 | 0 |
| bge_base_en | lancedb | 1335 | 39.196 | 52331 | 0 |
| bge_base_en | weaviate | 3789 | 13.81 | 52331 | 0 |
| gte_small | milvus | 3243 | 16.136 | 52331 | 0 |
| gte_small | qdrant | 2464 | 21.242 | 52331 | 0 |
| gte_small | chromadb | 608.1 | 86.052 | 52331 | 0 |
| gte_small | lancedb | 1349 | 38.784 | 52331 | 0 |
| gte_small | weaviate | 4656 | 11.24 | 52331 | 0 |
| gte_base | milvus | 3046 | 17.183 | 52331 | 0 |
| gte_base | qdrant | 1444 | 36.246 | 52331 | 0 |
| gte_base | chromadb | 590.8 | 88.571 | 52331 | 0 |
| gte_base | lancedb | 1353 | 38.671 | 52331 | 0 |
| gte_base | weaviate | 3322 | 15.753 | 52331 | 0 |
| e5_small_hf | milvus | 3145 | 16.64 | 52331 | 0 |
| e5_small_hf | qdrant | 2374 | 22.047 | 52331 | 0 |
| e5_small_hf | chromadb | 607.3 | 86.177 | 52331 | 0 |
| e5_small_hf | lancedb | 1338 | 39.125 | 52331 | 0 |
| e5_small_hf | weaviate | 4449 | 11.762 | 52331 | 0 |
| e5_base_hf | milvus | 3103 | 16.866 | 52331 | 0 |
| e5_base_hf | qdrant | 1380 | 37.911 | 52331 | 0 |
| e5_base_hf | chromadb | 573.9 | 91.178 | 52331 | 0 |
| e5_base_hf | lancedb | 1347 | 38.841 | 52331 | 0 |
| e5_base_hf | weaviate | 3519 | 14.872 | 52331 | 0 |

## Search QPS p99

_ARAMA QPS + p99 (max proxy) — TEST_RUNS=10 oldugu icin p99 ≈ max_time_  
| Sira | DB | Model | Algoritma | QPS | avg_ms | p50_ms | p95_ms | p99_proxy_ms | vram_peak_gb |
|---|---|---|---|---|---|---|---|---|---|
| 1 | chromadb | minilm_l12 | HNSW_batch | 3384 | 2.955 | 2.954 | 3.044 | 3.083 | 0 |
| 2 | chromadb | all_mini_l6_base | HNSW_batch | 3138 | 3.187 | 3.154 | 3.311 | 3.343 | 0 |
| 3 | chromadb | gte_small | HNSW_batch | 3060 | 3.268 | 3.227 | 3.485 | 3.52 | 0 |
| 4 | chromadb | bge_small_en | HNSW_batch | 2864 | 3.492 | 3.488 | 3.605 | 3.639 | 0 |
| 5 | chromadb | multi_qa_minilm | HNSW_batch | 2856 | 3.501 | 3.473 | 3.633 | 3.65 | 0 |
| 6 | chromadb | paraphrase_multi | HNSW_batch | 2835 | 3.527 | 3.518 | 3.62 | 3.635 | 0 |
| 7 | chromadb | e5_small_base | HNSW_batch | 2800 | 3.571 | 3.568 | 3.676 | 3.694 | 0 |
| 8 | chromadb | e5_small_hf | HNSW_batch | 2650 | 3.773 | 3.755 | 3.874 | 3.885 | 0 |
| 9 | chromadb | all_mini_l6 | HNSW_batch | 2626 | 3.808 | 3.774 | 3.952 | 3.954 | 0 |
| 10 | chromadb | e5_small | HNSW_batch | 2296 | 4.355 | 4.366 | 4.447 | 4.449 | 0 |
| 11 | chromadb | mpnet_base | HNSW_batch | 2208 | 4.528 | 4.504 | 4.792 | 4.913 | 0 |
| 12 | chromadb | distilroberta | HNSW_batch | 2144 | 4.665 | 4.648 | 4.845 | 4.87 | 0 |
| 13 | chromadb | bge_base_en | HNSW_batch | 1921 | 5.206 | 5.09 | 5.718 | 5.945 | 0 |
| 14 | chromadb | gte_base | HNSW_batch | 1920 | 5.207 | 5.176 | 5.33 | 5.349 | 0 |
| 15 | chromadb | mpnet_multi_base | HNSW_batch | 1874 | 5.337 | 5.31 | 5.453 | 5.483 | 0 |
| 16 | chromadb | multi_qa_mpnet | HNSW_batch | 1751 | 5.711 | 5.695 | 5.843 | 5.846 | 0 |
| 17 | chromadb | e5_base_hf | HNSW_batch | 1729 | 5.783 | 5.732 | 6.005 | 6.171 | 0 |
| 18 | chromadb | mpnet_multi | HNSW_batch | 1718 | 5.822 | 5.813 | 5.971 | 5.979 | 0 |
| 19 | chromadb | e5_base_base | HNSW_batch | 1618 | 6.182 | 6.222 | 6.409 | 6.49 | 0 |
| 20 | chromadb | snowflake_arctic_l_base | HNSW_batch | 1464 | 6.829 | 6.831 | 6.906 | 6.911 | 0 |
| 21 | chromadb | e5_base | HNSW_batch | 1423 | 7.029 | 7.024 | 7.209 | 7.252 | 0 |
| 22 | chromadb | bge_squad_base | HNSW_batch | 1408 | 7.104 | 7.02 | 7.613 | 8.012 | 0 |
| 23 | chromadb | bge_squad | HNSW_batch | 1331 | 7.511 | 7.126 | 9.093 | 9.429 | 0 |
| 24 | chromadb | snowflake_arctic_l | HNSW_batch | 1244 | 8.038 | 8.048 | 8.124 | 8.146 | 0 |
| 25 | milvus | e5_small | HNSW_batch | 1027 | 9.734 | 9.773 | 10.097 | 10.187 | 0 |
| 26 | milvus | paraphrase_multi | HNSW_batch | 1018 | 9.824 | 9.787 | 10.216 | 10.278 | 0 |
| 27 | milvus | all_mini_l6_base | HNSW_batch | 1006 | 9.945 | 9.858 | 10.451 | 10.577 | 0 |
| 28 | milvus | e5_small_hf | HNSW_batch | 1003 | 9.973 | 9.973 | 10.321 | 10.324 | 0 |
| 29 | milvus | e5_small_base | HNSW_batch | 998.1 | 10.019 | 9.881 | 10.53 | 10.648 | 0 |
| 30 | milvus | all_mini_l6 | HNSW_batch | 995.7 | 10.043 | 10.022 | 10.506 | 10.702 | 0 |
| 31 | milvus | gte_small | HNSW_batch | 985.2 | 10.15 | 10.078 | 11.091 | 11.49 | 0 |
| 32 | milvus | bge_small_en | HNSW_batch | 971.6 | 10.292 | 10.286 | 10.478 | 10.483 | 0 |
| 33 | milvus | multi_qa_minilm | HNSW_batch | 970.7 | 10.302 | 10.38 | 10.68 | 10.695 | 0 |
| 34 | milvus | minilm_l12 | HNSW_batch | 966.6 | 10.346 | 10.409 | 10.537 | 10.578 | 0 |
| 35 | milvus | e5_base_base | HNSW_batch | 607.9 | 16.451 | 16.377 | 16.931 | 17.253 | 0 |
| 36 | milvus | multi_qa_mpnet | HNSW_batch | 602.1 | 16.609 | 16.477 | 17.163 | 17.174 | 0 |
| 37 | milvus | bge_base_en | HNSW_batch | 595.2 | 16.8 | 16.624 | 17.602 | 17.846 | 0 |
| 38 | milvus | e5_base | HNSW_batch | 593.8 | 16.841 | 16.786 | 17.507 | 17.571 | 0 |
| 39 | milvus | mpnet_multi_base | HNSW_batch | 586.1 | 17.062 | 16.839 | 18.009 | 18.339 | 0 |
| 40 | milvus | e5_base_hf | HNSW_batch | 575.2 | 17.386 | 17.176 | 17.983 | 18.08 | 0 |
| 41 | milvus | mpnet_base | HNSW_batch | 568.5 | 17.591 | 17.722 | 18.376 | 18.472 | 0 |
| 42 | milvus | mpnet_multi | HNSW_batch | 565.1 | 17.696 | 17.425 | 18.864 | 19.322 | 0 |
| 43 | milvus | distilroberta | HNSW_batch | 561.5 | 17.811 | 17.024 | 22.52 | 26.755 | 0 |
| 44 | milvus | gte_base | HNSW_batch | 560.6 | 17.839 | 17.78 | 18.519 | 19.001 | 0 |
| 45 | milvus | snowflake_arctic_l_base | HNSW_batch | 456.5 | 21.906 | 21.622 | 23.061 | 23.185 | 0 |
| 46 | milvus | bge_squad_base | HNSW_batch | 449.6 | 22.244 | 21.949 | 24.315 | 24.527 | 0 |
| 47 | milvus | snowflake_arctic_l | HNSW_batch | 432.7 | 23.109 | 23.03 | 24.435 | 24.79 | 0 |
| 48 | milvus | bge_squad | HNSW_batch | 423.8 | 23.598 | 22.943 | 27.189 | 29.519 | 0 |
| 49 | chromadb | minilm_l12 | HNSW_n5 | 156.3 | 6.397 | 6.405 | 6.614 | 6.667 | 0 |
| 50 | chromadb | paraphrase_multi | HNSW_n5 | 147.1 | 6.798 | 6.813 | 6.871 | 6.884 | 0 |
| 51 | chromadb | gte_small | HNSW_n5 | 146.6 | 6.821 | 6.794 | 6.944 | 6.948 | 0 |
| 52 | chromadb | all_mini_l6_base | HNSW_n5 | 143.6 | 6.962 | 6.909 | 7.268 | 7.431 | 0 |
| 53 | chromadb | e5_small_base | HNSW_n5 | 140.4 | 7.121 | 7.128 | 7.215 | 7.249 | 0 |
| 54 | chromadb | multi_qa_minilm | HNSW_n5 | 135.7 | 7.371 | 7.373 | 7.608 | 7.652 | 0 |
| 55 | chromadb | multi_qa_minilm | HNSW_default | 132.1 | 7.571 | 7.507 | 8.361 | 8.598 | 0 |
| 56 | chromadb | bge_small_en | HNSW_n5 | 132 | 7.574 | 7.697 | 8.065 | 8.094 | 0 |
| 57 | chromadb | paraphrase_multi | HNSW_default | 131.3 | 7.614 | 7.661 | 8.174 | 8.181 | 0 |
| 58 | chromadb | gte_small | HNSW_default | 131.1 | 7.629 | 7.574 | 8.017 | 8.034 | 0 |
| 59 | chromadb | e5_small_hf | HNSW_n5 | 131 | 7.633 | 7.443 | 8.394 | 8.739 | 0 |
| 60 | chromadb | all_mini_l6_base | HNSW_default | 130.7 | 7.65 | 7.63 | 7.872 | 7.898 | 0 |
| 61 | chromadb | e5_small_base | HNSW_default | 128.8 | 7.766 | 7.715 | 8.026 | 8.026 | 0 |
| 62 | chromadb | e5_small | HNSW_n5 | 127.7 | 7.831 | 7.766 | 8.405 | 8.598 | 0 |
| 63 | chromadb | all_mini_l6 | HNSW_n5 | 126.8 | 7.887 | 7.85 | 8.099 | 8.125 | 0 |
| 64 | chromadb | minilm_l12 | HNSW_default | 125.9 | 7.944 | 7.721 | 9.618 | 10.155 | 0 |
| 65 | chromadb | bge_small_en | HNSW_default | 125.4 | 7.973 | 7.808 | 8.747 | 8.8 | 0 |
| 66 | weaviate | minilm_l12 | HNSW_limit5 | 125.3 | 7.979 | 7.787 | 8.958 | 9.641 | 0 |
| 67 | weaviate | all_mini_l6_base | HNSW_limit5 | 124 | 8.062 | 7.966 | 8.488 | 8.593 | 0 |
| 68 | chromadb | minilm_l12 | HNSW_n20 | 122.8 | 8.143 | 8.074 | 8.607 | 8.879 | 0 |
| 69 | weaviate | paraphrase_multi | HNSW_limit5 | 122.8 | 8.143 | 8.097 | 8.654 | 8.833 | 0 |
| 70 | chromadb | e5_small_hf | HNSW_default | 122.5 | 8.162 | 7.979 | 9.008 | 9.217 | 0 |
| 71 | chromadb | mpnet_base | HNSW_n5 | 121.7 | 8.214 | 8.09 | 8.924 | 9.413 | 0 |
| 72 | chromadb | distilroberta | HNSW_n5 | 121.4 | 8.24 | 8.156 | 8.728 | 9.004 | 0 |
| 73 | chromadb | e5_small | HNSW_default | 117.6 | 8.503 | 8.227 | 9.295 | 9.582 | 0 |
| 74 | chromadb | all_mini_l6 | HNSW_default | 116.3 | 8.597 | 8.323 | 9.879 | 10.643 | 0 |
| 75 | weaviate | bge_small_en | HNSW_limit5 | 116.1 | 8.61 | 8.484 | 9.035 | 9.07 | 0 |
| 76 | chromadb | bge_base_en | HNSW_n5 | 116.1 | 8.614 | 8.625 | 8.736 | 8.739 | 0 |
| 77 | chromadb | gte_base | HNSW_n5 | 115.5 | 8.658 | 8.607 | 9.022 | 9.1 | 0 |
| 78 | chromadb | gte_small | HNSW_n20 | 115 | 8.697 | 8.749 | 9.001 | 9.032 | 0 |
| 79 | weaviate | gte_small | HNSW_limit5 | 114.8 | 8.708 | 8.592 | 9.296 | 9.369 | 0 |
| 80 | weaviate | mpnet_base | HNSW_limit5 | 112.7 | 8.87 | 8.871 | 9.171 | 9.192 | 0 |
| 81 | weaviate | minilm_l12 | HNSW_default | 112.7 | 8.872 | 8.757 | 9.607 | 10.021 | 0 |
| 82 | chromadb | bge_small_en | HNSW_n20 | 112.4 | 8.9 | 8.899 | 9.015 | 9.027 | 0 |
| 83 | chromadb | distilroberta | HNSW_default | 111.8 | 8.947 | 8.885 | 9.426 | 9.628 | 0 |
| 84 | chromadb | mpnet_multi_base | HNSW_n5 | 111.4 | 8.975 | 8.96 | 9.137 | 9.171 | 0 |
| 85 | chromadb | multi_qa_minilm | HNSW_n20 | 110.6 | 9.042 | 9.037 | 9.319 | 9.405 | 0 |
| 86 | chromadb | mpnet_base | HNSW_default | 110.2 | 9.078 | 9.146 | 9.319 | 9.324 | 0 |
| 87 | chromadb | bge_base_en | HNSW_default | 109.9 | 9.1 | 9.154 | 9.252 | 9.274 | 0 |
| 88 | chromadb | all_mini_l6_base | HNSW_n20 | 109.5 | 9.136 | 8.908 | 10.411 | 11.357 | 0 |
| 89 | chromadb | e5_base_base | HNSW_n5 | 109.5 | 9.136 | 9.161 | 9.362 | 9.422 | 0 |
| 90 | weaviate | all_mini_l6_base | HNSW_default | 109.3 | 9.147 | 8.839 | 10.603 | 11.003 | 0 |
| 91 | chromadb | all_mini_l6 | HNSW_n20 | 108.3 | 9.237 | 9.293 | 9.595 | 9.681 | 0 |
| 92 | chromadb | mpnet_multi | HNSW_n5 | 107.9 | 9.272 | 9.273 | 9.491 | 9.593 | 0 |
| 93 | chromadb | gte_base | HNSW_default | 107.2 | 9.326 | 9.265 | 9.788 | 10.01 | 0 |
| 94 | chromadb | paraphrase_multi | HNSW_n20 | 106.7 | 9.371 | 9.23 | 10.164 | 10.742 | 0 |
| 95 | weaviate | e5_small_base | HNSW_limit5 | 106.1 | 9.424 | 9.317 | 9.997 | 10.246 | 0 |
| 96 | chromadb | e5_base_hf | HNSW_n5 | 106 | 9.43 | 9.403 | 9.746 | 9.772 | 0 |
| 97 | chromadb | multi_qa_mpnet | HNSW_n5 | 105.8 | 9.45 | 9.462 | 9.616 | 9.639 | 0 |
| 98 | weaviate | distilroberta | HNSW_limit5 | 105.8 | 9.45 | 9.302 | 9.944 | 10.195 | 0 |
| 99 | chromadb | e5_small_base | HNSW_n20 | 105.8 | 9.453 | 9.257 | 10.369 | 10.873 | 0 |
| 100 | weaviate | e5_small_hf | HNSW_limit5 | 105.7 | 9.461 | 9.461 | 9.711 | 9.722 | 0 |
| 101 | chromadb | e5_small | HNSW_n20 | 104.3 | 9.587 | 9.363 | 10.208 | 10.296 | 0 |
| 102 | chromadb | mpnet_multi_base | HNSW_default | 103.3 | 9.678 | 9.631 | 9.997 | 10.139 | 0 |
| 103 | weaviate | bge_base_en | HNSW_limit5 | 103.1 | 9.697 | 9.753 | 9.961 | 9.987 | 0 |
| 104 | weaviate | paraphrase_multi | HNSW_default | 102.4 | 9.761 | 9.651 | 10.368 | 10.371 | 0 |
| 105 | weaviate | bge_small_en | BM25 | 102.3 | 9.779 | 9.764 | 10.022 | 10.084 | 0 |
| 106 | chromadb | e5_base_base | HNSW_default | 102.2 | 9.784 | 9.786 | 9.983 | 10.001 | 0 |
| 107 | weaviate | multi_qa_minilm | HNSW_limit5 | 102.1 | 9.793 | 9.709 | 10.307 | 10.478 | 0 |
| 108 | chromadb | e5_small_hf | HNSW_n20 | 101.8 | 9.823 | 9.855 | 10.113 | 10.118 | 0 |
| 109 | weaviate | gte_small | HNSW_default | 100.9 | 9.912 | 9.708 | 11.045 | 11.277 | 0 |
| 110 | weaviate | mpnet_base | HNSW_default | 100.7 | 9.927 | 9.723 | 11.046 | 11.77 | 0 |
| 111 | weaviate | e5_small_hf | BM25 | 100.7 | 9.929 | 9.865 | 10.445 | 10.719 | 0 |
| 112 | weaviate | gte_base | BM25 | 100.7 | 9.933 | 9.888 | 10.279 | 10.408 | 0 |
| 113 | weaviate | gte_small | BM25 | 100.4 | 9.958 | 9.898 | 10.264 | 10.362 | 0 |
| 114 | weaviate | distilroberta | BM25 | 100.4 | 9.964 | 9.858 | 10.309 | 10.338 | 0 |
| 115 | weaviate | e5_base_hf | BM25 | 100.2 | 9.984 | 9.948 | 10.289 | 10.298 | 0 |
| 116 | weaviate | multi_qa_minilm | BM25 | 100.1 | 9.986 | 9.936 | 10.355 | 10.606 | 0 |
| 117 | chromadb | bge_squad_base | HNSW_n5 | 100.1 | 9.987 | 9.996 | 10.237 | 10.311 | 0 |
| 118 | weaviate | multi_qa_mpnet | BM25 | 100.1 | 9.993 | 9.974 | 10.368 | 10.463 | 0 |
| 119 | weaviate | bge_base_en | BM25 | 100.1 | 9.994 | 9.916 | 10.247 | 10.249 | 0 |
| 120 | weaviate | e5_small | BM25 | 99.9 | 10.012 | 9.906 | 10.764 | 11.074 | 0 |
| 121 | weaviate | snowflake_arctic_l_base | BM25 | 99.8 | 10.016 | 9.965 | 10.47 | 10.814 | 0 |
| 122 | chromadb | mpnet_multi | HNSW_default | 99.6 | 10.036 | 9.96 | 10.439 | 10.721 | 0 |
| 123 | weaviate | paraphrase_multi | BM25 | 99.6 | 10.038 | 9.941 | 10.389 | 10.425 | 0 |
| 124 | weaviate | bge_small_en | HNSW_default | 99.4 | 10.06 | 9.982 | 10.687 | 11.02 | 0 |
| 125 | weaviate | bge_squad_base | BM25 | 99 | 10.105 | 10.102 | 10.413 | 10.618 | 0 |
| 126 | weaviate | distilroberta | HNSW_default | 98.9 | 10.109 | 9.781 | 11.395 | 12.249 | 0 |
| 127 | weaviate | minilm_l12 | BM25 | 98.5 | 10.15 | 10.058 | 10.525 | 10.684 | 0 |
| 128 | chromadb | e5_base | HNSW_n5 | 98.5 | 10.154 | 10.216 | 10.353 | 10.366 | 0 |
| 129 | chromadb | e5_base_hf | HNSW_default | 98.3 | 10.171 | 10.143 | 10.543 | 10.589 | 0 |
| 130 | weaviate | gte_base | HNSW_limit5 | 98.2 | 10.188 | 10.257 | 10.59 | 10.825 | 0 |
| 131 | weaviate | e5_base_base | BM25 | 97.8 | 10.221 | 10.027 | 11.113 | 11.235 | 0 |
| 132 | weaviate | e5_small_base | BM25 | 97.4 | 10.271 | 10.196 | 10.692 | 10.769 | 0 |
| 133 | weaviate | bge_squad | BM25 | 97.2 | 10.287 | 10.256 | 10.564 | 10.591 | 0 |
| 134 | chromadb | multi_qa_mpnet | HNSW_default | 97 | 10.304 | 10.316 | 10.577 | 10.674 | 0 |
| 135 | weaviate | all_mini_l6_base | BM25 | 96.8 | 10.329 | 10.277 | 10.722 | 10.879 | 0 |
| 136 | weaviate | mpnet_base | BM25 | 96.8 | 10.329 | 10.2 | 11.099 | 11.651 | 0 |
| 137 | chromadb | distilroberta | HNSW_n20 | 96.8 | 10.332 | 10.279 | 10.775 | 10.898 | 0 |
| 138 | chromadb | mpnet_base | HNSW_n20 | 96.6 | 10.347 | 10.362 | 10.483 | 10.502 | 0 |
| 139 | weaviate | mpnet_multi_base | HNSW_limit5 | 96.2 | 10.4 | 10.182 | 11.322 | 11.843 | 0 |
| 140 | weaviate | snowflake_arctic_l | BM25 | 95.8 | 10.435 | 10.355 | 10.906 | 11.008 | 0 |
| 141 | chromadb | gte_base | HNSW_n20 | 94.6 | 10.573 | 10.552 | 10.952 | 10.971 | 0 |
| 142 | weaviate | e5_base | BM25 | 94.6 | 10.575 | 10.512 | 11.07 | 11.2 | 0 |
| 143 | chromadb | bge_base_en | HNSW_n20 | 93.7 | 10.671 | 10.667 | 10.917 | 10.936 | 0 |
| 144 | weaviate | mpnet_multi_base | BM25 | 93.5 | 10.694 | 10.492 | 11.408 | 11.735 | 0 |
| 145 | weaviate | multi_qa_mpnet | HNSW_limit5 | 92.7 | 10.782 | 10.755 | 11.163 | 11.333 | 0 |
| 146 | chromadb | snowflake_arctic_l_base | HNSW_n5 | 92.6 | 10.802 | 10.803 | 11.175 | 11.277 | 0 |
| 147 | weaviate | all_mini_l6 | HNSW_limit5 | 92.4 | 10.828 | 10.8 | 11.626 | 12.049 | 0 |
| 148 | weaviate | e5_small_hf | HNSW_default | 92.3 | 10.83 | 10.708 | 11.637 | 11.771 | 0 |
| 149 | weaviate | all_mini_l6 | BM25 | 92.3 | 10.839 | 10.697 | 11.676 | 11.846 | 0 |
| 150 | chromadb | bge_squad | HNSW_default | 91.7 | 10.906 | 11.021 | 11.273 | 11.409 | 0 |
| 151 | weaviate | mpnet_multi | BM25 | 90.8 | 11.014 | 10.533 | 12.358 | 12.665 | 0 |
| 152 | chromadb | bge_squad | HNSW_n5 | 90.4 | 11.068 | 10.695 | 12.688 | 13.387 | 0 |
| 153 | weaviate | e5_small_base | HNSW_default | 90.3 | 11.073 | 10.935 | 11.724 | 11.931 | 0 |
| 154 | chromadb | mpnet_multi_base | HNSW_n20 | 90.2 | 11.082 | 11.022 | 11.47 | 11.644 | 0 |
| 155 | weaviate | bge_base_en | HNSW_default | 90.2 | 11.092 | 10.876 | 12.026 | 12.136 | 0 |
| 156 | chromadb | bge_squad_base | HNSW_default | 90 | 11.108 | 10.934 | 11.834 | 11.917 | 0 |
| 157 | chromadb | snowflake_arctic_l_base | HNSW_default | 89.4 | 11.191 | 11.333 | 11.572 | 11.611 | 0 |
| 158 | chromadb | e5_base_base | HNSW_n20 | 89.3 | 11.202 | 11.175 | 11.558 | 11.597 | 0 |
| 159 | chromadb | e5_base_hf | HNSW_n20 | 88.9 | 11.251 | 11.238 | 11.573 | 11.606 | 0 |
| 160 | chromadb | snowflake_arctic_l | HNSW_n5 | 88.8 | 11.264 | 11.263 | 11.527 | 11.534 | 0 |
| 161 | chromadb | mpnet_multi | HNSW_n20 | 88.3 | 11.32 | 11.229 | 11.58 | 11.6 | 0 |
| 162 | weaviate | snowflake_arctic_l_base | HNSW_limit5 | 88.2 | 11.336 | 11.349 | 11.783 | 11.965 | 0 |
| 163 | weaviate | e5_base_hf | HNSW_limit5 | 88 | 11.362 | 11.32 | 11.64 | 11.714 | 0 |
| 164 | weaviate | e5_base_base | HNSW_limit5 | 87.8 | 11.386 | 11.218 | 12.346 | 13.033 | 0 |
| 165 | chromadb | e5_base | HNSW_default | 87.6 | 11.419 | 10.935 | 13.109 | 13.407 | 0 |
| 166 | weaviate | gte_base | HNSW_default | 87.4 | 11.447 | 11.317 | 12.058 | 12.24 | 0 |
| 167 | chromadb | minilm_l12 | HNSW_n50 | 86.6 | 11.552 | 11.24 | 12.753 | 12.769 | 0 |
| 168 | weaviate | e5_small | HNSW_limit5 | 86 | 11.628 | 11.694 | 12.055 | 12.209 | 0 |
| 169 | chromadb | all_mini_l6_base | HNSW_n50 | 85.4 | 11.713 | 11.716 | 11.865 | 11.93 | 0 |
| 170 | chromadb | gte_small | HNSW_n50 | 85 | 11.762 | 11.76 | 12.026 | 12.069 | 0 |
| 171 | weaviate | bge_squad_base | HNSW_limit5 | 84 | 11.901 | 11.606 | 13.053 | 13.315 | 0 |
| 172 | weaviate | all_mini_l6 | HNSW_default | 83.8 | 11.93 | 11.882 | 12.851 | 13.052 | 0 |
| 173 | chromadb | multi_qa_mpnet | HNSW_n20 | 83.8 | 11.934 | 11.955 | 12.278 | 12.326 | 0 |
| 174 | chromadb | all_mini_l6 | HNSW_n50 | 83.8 | 11.935 | 11.858 | 12.35 | 12.583 | 0 |
| 175 | chromadb | bge_squad | HNSW_n20 | 83.6 | 11.963 | 11.974 | 12.15 | 12.166 | 0 |
| 176 | chromadb | bge_small_en | HNSW_n50 | 83.2 | 12.014 | 11.901 | 12.485 | 12.562 | 0 |
| 177 | chromadb | e5_small_base | HNSW_n50 | 82.3 | 12.146 | 11.923 | 12.846 | 12.939 | 0 |
| 178 | weaviate | bge_squad | HNSW_limit5 | 82.3 | 12.155 | 11.994 | 12.999 | 13.131 | 0 |
| 179 | weaviate | multi_qa_mpnet | HNSW_default | 82.2 | 12.164 | 11.851 | 13.357 | 13.402 | 0 |
| 180 | chromadb | bge_squad_base | HNSW_n20 | 82 | 12.189 | 12.256 | 12.431 | 12.448 | 0 |
| 181 | chromadb | snowflake_arctic_l | HNSW_default | 81.8 | 12.222 | 12.239 | 12.414 | 12.514 | 0 |
| 182 | weaviate | mpnet_multi | HNSW_limit5 | 81.1 | 12.324 | 12.28 | 13.09 | 13.198 | 0 |
| 183 | weaviate | mpnet_multi_base | HNSW_default | 80.6 | 12.414 | 12.288 | 13.121 | 13.336 | 0 |
| 184 | chromadb | e5_base | HNSW_n20 | 80.5 | 12.423 | 12.472 | 12.626 | 12.7 | 0 |
| 185 | weaviate | e5_base_hf | HNSW_default | 80.3 | 12.461 | 12.376 | 12.963 | 13.206 | 0 |
| 186 | weaviate | e5_base_base | HNSW_default | 80.1 | 12.488 | 12.293 | 13.387 | 13.771 | 0 |
| 187 | weaviate | all_mini_l6_base | HNSW_limit20 | 78.2 | 12.795 | 12.686 | 13.683 | 14.048 | 0 |
| 188 | weaviate | snowflake_arctic_l_base | HNSW_default | 77.9 | 12.839 | 12.663 | 13.603 | 13.787 | 0 |
| 189 | weaviate | mpnet_multi | HNSW_default | 77 | 12.993 | 12.747 | 14.244 | 14.692 | 0 |
| 190 | chromadb | e5_small_hf | HNSW_n50 | 76.9 | 13.005 | 12.941 | 13.431 | 13.532 | 0 |
| 191 | weaviate | bge_squad | HNSW_default | 76.7 | 13.039 | 12.956 | 13.455 | 13.575 | 0 |
| 192 | weaviate | bge_squad_base | HNSW_default | 76.6 | 13.059 | 13.149 | 13.735 | 13.764 | 0 |
| 193 | chromadb | snowflake_arctic_l | HNSW_n20 | 76.3 | 13.1 | 12.991 | 13.873 | 14.05 | 0 |
| 194 | weaviate | snowflake_arctic_l | HNSW_limit5 | 76 | 13.166 | 13.062 | 13.772 | 13.831 | 0 |
| 195 | chromadb | multi_qa_minilm | HNSW_n50 | 75.8 | 13.191 | 13.176 | 13.766 | 13.779 | 0 |
| 196 | chromadb | paraphrase_multi | HNSW_n50 | 75.4 | 13.256 | 13.1 | 14.285 | 15.108 | 0 |
| 197 | chromadb | e5_small | HNSW_n50 | 74.4 | 13.433 | 13.372 | 13.733 | 13.806 | 0 |
| 198 | chromadb | snowflake_arctic_l_base | HNSW_n20 | 74 | 13.511 | 13.319 | 14.746 | 15.14 | 0 |
| 199 | weaviate | minilm_l12 | HNSW_limit20 | 73.5 | 13.599 | 13.405 | 14.57 | 14.874 | 0 |
| 200 | weaviate | bge_small_en | HNSW_limit20 | 72.9 | 13.722 | 13.619 | 14.194 | 14.252 | 0 |
| 201 | weaviate | paraphrase_multi | HNSW_limit20 | 72.8 | 13.745 | 13.466 | 14.953 | 15.648 | 0 |
| 202 | weaviate | e5_small | HNSW_default | 71.2 | 14.046 | 14.019 | 15.219 | 15.712 | 0 |
| 203 | weaviate | distilroberta | HNSW_limit20 | 70.5 | 14.179 | 13.811 | 15.3 | 15.514 | 0 |
| 204 | chromadb | distilroberta | HNSW_n50 | 70.4 | 14.196 | 13.871 | 15.694 | 16.522 | 0 |
| 205 | chromadb | mpnet_base | HNSW_n50 | 70.4 | 14.198 | 14.217 | 14.428 | 14.46 | 0 |
| 206 | chromadb | bge_base_en | HNSW_n50 | 69.6 | 14.362 | 14.262 | 15.255 | 15.971 | 0 |
| 207 | weaviate | e5_base | HNSW_limit5 | 69.4 | 14.402 | 14.257 | 15.52 | 15.821 | 0 |
| 208 | chromadb | gte_base | HNSW_n50 | 69.3 | 14.42 | 14.375 | 15.752 | 16.226 | 0 |
| 209 | weaviate | mpnet_base | HNSW_limit20 | 68.9 | 14.512 | 14.565 | 14.939 | 14.992 | 0 |
| 210 | weaviate | multi_qa_minilm | HNSW_default | 67.7 | 14.763 | 12.455 | 25.655 | 26.618 | 0 |
| 211 | chromadb | mpnet_multi_base | HNSW_n50 | 67.6 | 14.797 | 14.834 | 15.027 | 15.091 | 0 |
| 212 | weaviate | snowflake_arctic_l | HNSW_default | 66.9 | 14.939 | 14.805 | 16.143 | 16.514 | 0 |
| 213 | chromadb | bge_squad | HNSW_n50 | 65.4 | 15.3 | 15.316 | 15.472 | 15.532 | 0 |
| 214 | chromadb | mpnet_multi | HNSW_n50 | 65.3 | 15.307 | 15.209 | 15.691 | 15.863 | 0 |
| 215 | chromadb | e5_base_hf | HNSW_n50 | 65.1 | 15.35 | 15.347 | 15.518 | 15.547 | 0 |
| 216 | chromadb | multi_qa_mpnet | HNSW_n50 | 64.6 | 15.487 | 15.502 | 15.78 | 15.852 | 0 |
| 217 | chromadb | e5_base_base | HNSW_n50 | 64.3 | 15.551 | 15.445 | 16.136 | 16.333 | 0 |
| 218 | weaviate | e5_base | HNSW_default | 64.3 | 15.551 | 15.428 | 16.421 | 16.473 | 0 |
| 219 | weaviate | gte_small | HNSW_limit20 | 63.8 | 15.672 | 16.02 | 16.479 | 16.5 | 0 |
| 220 | weaviate | multi_qa_minilm | HNSW_limit20 | 62.8 | 15.92 | 16.112 | 16.624 | 16.625 | 0 |
| 221 | chromadb | snowflake_arctic_l_base | HNSW_n50 | 62.2 | 16.081 | 16.042 | 16.287 | 16.306 | 0 |
| 222 | weaviate | bge_base_en | HNSW_limit20 | 61.9 | 16.151 | 16.028 | 16.688 | 16.69 | 0 |
| 223 | chromadb | e5_base | HNSW_n50 | 61 | 16.401 | 16.286 | 17.118 | 17.218 | 0 |
| 224 | chromadb | bge_squad_base | HNSW_n50 | 60.9 | 16.41 | 16.419 | 17.409 | 17.753 | 0 |
| 225 | weaviate | mpnet_multi_base | HNSW_limit20 | 60.5 | 16.54 | 16.207 | 18.44 | 19.411 | 0 |
| 226 | weaviate | e5_small_hf | HNSW_limit20 | 60.4 | 16.556 | 16.578 | 16.938 | 16.947 | 0 |
| 227 | weaviate | e5_small_base | HNSW_limit20 | 59.8 | 16.733 | 16.611 | 18.683 | 19.262 | 0 |
| 228 | weaviate | gte_base | HNSW_limit20 | 58.8 | 17.009 | 16.911 | 17.812 | 18.046 | 0 |
| 229 | chromadb | snowflake_arctic_l | HNSW_n50 | 58 | 17.253 | 16.814 | 18.892 | 19.153 | 0 |
| 230 | weaviate | multi_qa_mpnet | HNSW_limit20 | 57 | 17.548 | 17.634 | 18.084 | 18.217 | 0 |
| 231 | weaviate | all_mini_l6 | HNSW_limit20 | 55.1 | 18.16 | 18.232 | 19.44 | 19.719 | 0 |
| 232 | weaviate | e5_base_base | HNSW_limit20 | 53.4 | 18.721 | 18.521 | 20.287 | 20.693 | 0 |
| 233 | weaviate | bge_squad_base | HNSW_limit20 | 52.8 | 18.944 | 18.921 | 20.11 | 20.756 | 0 |
| 234 | weaviate | e5_small | HNSW_limit20 | 52.7 | 18.982 | 18.674 | 20.742 | 21.221 | 0 |
| 235 | weaviate | snowflake_arctic_l_base | HNSW_limit20 | 52.6 | 19.012 | 18.44 | 20.333 | 20.411 | 0 |
| 236 | weaviate | e5_base_hf | HNSW_limit20 | 52.4 | 19.067 | 18.852 | 20.301 | 20.95 | 0 |
| 237 | weaviate | bge_squad | HNSW_limit20 | 51.3 | 19.48 | 19.513 | 20.307 | 20.658 | 0 |
| 238 | qdrant | mpnet_multi | HNSW_ef32 | 51.1 | 19.555 | 19.493 | 20.085 | 20.184 | 0 |
| 239 | weaviate | mpnet_multi | HNSW_limit20 | 49.1 | 20.364 | 19.744 | 22.049 | 22.137 | 0 |
| 240 | qdrant | gte_base | HNSW_ef32 | 49 | 20.389 | 20.477 | 20.633 | 20.651 | 0 |
| 241 | qdrant | mpnet_base | HNSW_ef32 | 48.7 | 20.528 | 20.529 | 21.032 | 21.184 | 0 |
| 242 | qdrant | e5_base_base | HNSW_ef32 | 47.9 | 20.864 | 20.68 | 21.685 | 21.98 | 0 |
| 243 | weaviate | snowflake_arctic_l | HNSW_limit20 | 46.8 | 21.352 | 21.333 | 22.572 | 23.084 | 0 |
| 244 | qdrant | bge_base_en | HNSW_ef32 | 46.3 | 21.606 | 21.676 | 21.818 | 21.855 | 0 |
| 245 | qdrant | distilroberta | HNSW_ef32 | 46.2 | 21.631 | 21.609 | 22.188 | 22.196 | 0 |
| 246 | qdrant | bge_squad_base | HNSW_ef32 | 46 | 21.73 | 21.703 | 22.251 | 22.4 | 0 |
| 247 | qdrant | snowflake_arctic_l_base | HNSW_ef32 | 44.8 | 22.332 | 22.143 | 23.185 | 23.257 | 0 |
| 248 | qdrant | e5_base_hf | HNSW_ef32 | 44.5 | 22.456 | 22.411 | 22.778 | 22.815 | 0 |
| 249 | qdrant | mpnet_multi_base | HNSW_ef32 | 44.4 | 22.545 | 22.432 | 22.906 | 23.083 | 0 |
| 250 | weaviate | e5_base | HNSW_limit20 | 44.2 | 22.62 | 22.628 | 22.869 | 22.892 | 0 |
| 251 | qdrant | bge_squad | HNSW_ef32 | 43.9 | 22.791 | 22.82 | 23.162 | 23.244 | 0 |
| 252 | qdrant | multi_qa_mpnet | HNSW_ef32 | 43.4 | 23.034 | 23.028 | 23.216 | 23.289 | 0 |
| 253 | qdrant | snowflake_arctic_l | HNSW_ef32 | 42.7 | 23.441 | 23.442 | 24.003 | 24.128 | 0 |
| 254 | qdrant | e5_base | HNSW_ef32 | 41.5 | 24.096 | 23.977 | 24.497 | 24.535 | 0 |
| 255 | qdrant | mpnet_base | HNSW_default | 40.4 | 24.734 | 24.502 | 25.993 | 26.424 | 0 |
| 256 | qdrant | mpnet_multi | HNSW_default | 40.1 | 24.96 | 24.89 | 25.511 | 25.536 | 0 |
| 257 | qdrant | distilroberta | HNSW_default | 40.1 | 24.96 | 24.852 | 26.17 | 26.747 | 0 |
| 258 | qdrant | mpnet_base | HNSW_ef128 | 39.9 | 25.044 | 24.981 | 25.476 | 25.749 | 0 |
| 259 | weaviate | all_mini_l6_base | HNSW_limit50 | 39.5 | 25.32 | 25.27 | 26.264 | 26.365 | 0 |
| 260 | qdrant | gte_base | HNSW_default | 39.3 | 25.468 | 25.368 | 25.922 | 26.028 | 0 |
| 261 | qdrant | e5_base_base | HNSW_default | 39 | 25.612 | 25.566 | 26.164 | 26.237 | 0 |
| 262 | qdrant | gte_base | HNSW_ef128 | 38.7 | 25.81 | 25.908 | 26.708 | 26.729 | 0 |
| 263 | qdrant | bge_base_en | HNSW_default | 38.5 | 25.956 | 25.878 | 26.777 | 26.849 | 0 |
| 264 | qdrant | distilroberta | HNSW_ef128 | 38.5 | 25.997 | 26.067 | 26.444 | 26.465 | 0 |
| 265 | weaviate | minilm_l12 | HNSW_limit50 | 38.3 | 26.127 | 26.028 | 27.337 | 28.169 | 0 |
| 266 | qdrant | mpnet_multi | HNSW_ef128 | 38.3 | 26.142 | 26.237 | 26.922 | 27.083 | 0 |
| 267 | qdrant | multi_qa_mpnet | HNSW_default | 38 | 26.282 | 26.281 | 26.484 | 26.563 | 0 |
| 268 | qdrant | mpnet_multi_base | HNSW_default | 37.7 | 26.513 | 26.43 | 26.903 | 26.982 | 0 |
| 269 | qdrant | bge_base_en | HNSW_ef128 | 37.7 | 26.523 | 26.416 | 27.273 | 27.491 | 0 |
| 270 | weaviate | minilm_l12 | HYBRID_alpha0.5 | 37.6 | 26.57 | 26.383 | 27.439 | 27.801 | 0 |
| 271 | weaviate | all_mini_l6_base | HYBRID_alpha0.5 | 37.5 | 26.666 | 26.536 | 27.219 | 27.428 | 0 |
| 272 | weaviate | minilm_l12 | HYBRID_alpha0.75 | 37.5 | 26.7 | 26.513 | 27.436 | 27.732 | 0 |
| 273 | weaviate | all_mini_l6_base | HYBRID_alpha0.75 | 37.4 | 26.751 | 26.602 | 27.312 | 27.537 | 0 |
| 274 | qdrant | e5_base_base | HNSW_ef128 | 37.2 | 26.913 | 26.928 | 27.728 | 27.928 | 0 |
| 275 | qdrant | bge_squad_base | HNSW_default | 37 | 27.009 | 26.943 | 27.418 | 27.551 | 0 |
| 276 | qdrant | mpnet_multi_base | HNSW_ef128 | 36.4 | 27.466 | 27.364 | 28.203 | 28.522 | 0 |
| 277 | weaviate | all_mini_l6_base | HYBRID_alpha0.25 | 36.3 | 27.518 | 26.775 | 29.558 | 29.899 | 0 |
| 278 | weaviate | mpnet_base | HNSW_limit50 | 36.3 | 27.521 | 27.319 | 28.463 | 28.974 | 0 |
| 279 | weaviate | paraphrase_multi | HNSW_limit50 | 36.3 | 27.546 | 27.605 | 28.055 | 28.157 | 0 |
| 280 | qdrant | snowflake_arctic_l_base | HNSW_default | 36.1 | 27.681 | 27.527 | 29.485 | 30.162 | 0 |
| 281 | qdrant | multi_qa_mpnet | HNSW_ef128 | 36 | 27.778 | 27.78 | 28.291 | 28.359 | 0 |
| 282 | qdrant | bge_squad | HNSW_default | 35.9 | 27.831 | 27.85 | 28.117 | 28.211 | 0 |
| 283 | weaviate | minilm_l12 | HYBRID_alpha0.25 | 35.8 | 27.917 | 27.738 | 29.666 | 29.68 | 0 |
| 284 | weaviate | bge_small_en | HNSW_limit50 | 35.4 | 28.243 | 28.278 | 28.753 | 28.9 | 0 |
| 285 | weaviate | distilroberta | HNSW_limit50 | 35.4 | 28.244 | 28.163 | 28.992 | 29.181 | 0 |
| 286 | weaviate | gte_small | HNSW_limit50 | 35.2 | 28.439 | 28.416 | 28.851 | 28.881 | 0 |
| 287 | qdrant | bge_squad_base | HNSW_ef128 | 35 | 28.563 | 28.556 | 28.904 | 28.965 | 0 |
| 288 | weaviate | paraphrase_multi | HYBRID_alpha0.5 | 35 | 28.57 | 28.558 | 28.832 | 28.859 | 0 |
| 289 | weaviate | paraphrase_multi | HYBRID_alpha0.75 | 35 | 28.595 | 28.535 | 29.104 | 29.209 | 0 |
| 290 | weaviate | mpnet_base | HYBRID_alpha0.5 | 34.8 | 28.704 | 28.634 | 29.351 | 29.474 | 0 |
| 291 | qdrant | e5_base_hf | HNSW_ef128 | 34.7 | 28.778 | 28.779 | 29.141 | 29.174 | 0 |
| 292 | weaviate | mpnet_base | HYBRID_alpha0.75 | 34.7 | 28.818 | 28.845 | 29.487 | 29.794 | 0 |
| 293 | qdrant | mpnet_base | HNSW_ef256 | 34.7 | 28.84 | 28.822 | 29.081 | 29.081 | 0 |
| 294 | qdrant | snowflake_arctic_l_base | HNSW_ef128 | 34.6 | 28.886 | 28.542 | 30.584 | 31.112 | 0 |
| 295 | weaviate | paraphrase_multi | HYBRID_alpha0.25 | 34.3 | 29.177 | 28.954 | 30.352 | 30.46 | 0 |
| 296 | weaviate | bge_base_en | HNSW_limit50 | 34.2 | 29.258 | 29.076 | 30.629 | 31.672 | 0 |
| 297 | weaviate | bge_small_en | HYBRID_alpha0.5 | 33.9 | 29.476 | 29.528 | 29.738 | 29.761 | 0 |
| 298 | qdrant | snowflake_arctic_l | HNSW_default | 33.7 | 29.649 | 29.703 | 29.959 | 30.037 | 0 |
| 299 | qdrant | e5_base | HNSW_default | 33.7 | 29.695 | 29.614 | 30.239 | 30.24 | 0 |
| 300 | weaviate | multi_qa_minilm | HNSW_limit50 | 33.6 | 29.803 | 29.859 | 30.079 | 30.164 | 0 |
| 301 | weaviate | bge_small_en | HYBRID_alpha0.75 | 33.5 | 29.867 | 29.377 | 32.337 | 34.386 | 0 |
| 302 | qdrant | bge_squad | HNSW_ef128 | 33.5 | 29.876 | 29.546 | 31.487 | 32.326 | 0 |
| 303 | weaviate | mpnet_base | HYBRID_alpha0.25 | 33.3 | 30.043 | 29.326 | 33.336 | 34.151 | 0 |
| 304 | weaviate | distilroberta | HYBRID_alpha0.75 | 33.2 | 30.121 | 29.814 | 31.344 | 31.894 | 0 |
| 305 | weaviate | distilroberta | HYBRID_alpha0.5 | 33.1 | 30.195 | 30.334 | 31.022 | 31.148 | 0 |
| 306 | qdrant | distilroberta | HNSW_ef256 | 33 | 30.3 | 30.063 | 31.231 | 31.375 | 0 |
| 307 | qdrant | paraphrase_multi | HNSW_ef128 | 32.9 | 30.412 | 30.375 | 30.916 | 30.917 | 0 |
| 308 | qdrant | paraphrase_multi | HNSW_ef256 | 32.9 | 30.436 | 30.421 | 30.864 | 31.088 | 0 |
| 309 | weaviate | gte_small | HYBRID_alpha0.5 | 32.7 | 30.571 | 30.609 | 30.895 | 30.935 | 0 |
| 310 | weaviate | bge_small_en | HYBRID_alpha0.25 | 32.6 | 30.666 | 30.508 | 32.005 | 32.092 | 0 |
| 311 | qdrant | paraphrase_multi | HNSW_ef32 | 32.4 | 30.885 | 30.729 | 31.613 | 31.922 | 0 |
| 312 | weaviate | multi_qa_minilm | HYBRID_alpha0.75 | 32.3 | 30.919 | 30.996 | 31.088 | 31.116 | 0 |
| 313 | weaviate | multi_qa_minilm | HYBRID_alpha0.5 | 32.2 | 31.056 | 30.991 | 31.334 | 31.344 | 0 |
| 314 | qdrant | bge_base_en | HNSW_ef256 | 32.2 | 31.075 | 31.133 | 31.484 | 31.603 | 0 |
| 315 | qdrant | bge_small_en | HNSW_ef256 | 32.1 | 31.12 | 31.013 | 31.747 | 31.983 | 0 |
| 316 | qdrant | all_mini_l6 | HNSW_default | 32.1 | 31.183 | 31.236 | 31.942 | 31.989 | 0 |
| 317 | qdrant | minilm_l12 | HNSW_ef128 | 32 | 31.262 | 31.261 | 31.81 | 31.856 | 0 |
| 318 | weaviate | e5_small_hf | HNSW_limit50 | 31.9 | 31.311 | 31.362 | 31.68 | 31.706 | 0 |
| 319 | weaviate | e5_small_base | HNSW_limit50 | 31.9 | 31.318 | 31.256 | 32.655 | 33.088 | 0 |
| 320 | qdrant | minilm_l12 | HNSW_ef256 | 31.9 | 31.329 | 31.234 | 32.002 | 32.325 | 0 |
| 321 | qdrant | paraphrase_multi | EXACT | 31.9 | 31.356 | 31.435 | 31.853 | 31.858 | 0 |
| 322 | qdrant | e5_small_base | HNSW_ef256 | 31.8 | 31.407 | 31.326 | 31.836 | 31.986 | 0 |
| 323 | qdrant | e5_small | HNSW_ef32 | 31.8 | 31.472 | 31.434 | 32.667 | 32.703 | 0 |
| 324 | qdrant | minilm_l12 | HNSW_ef32 | 31.7 | 31.522 | 31.573 | 32.04 | 32.118 | 0 |
| 325 | weaviate | gte_small | HYBRID_alpha0.25 | 31.6 | 31.625 | 31.217 | 33.527 | 33.583 | 0 |
| 326 | qdrant | bge_small_en | HNSW_ef128 | 31.6 | 31.662 | 31.636 | 32.216 | 32.227 | 0 |
| 327 | weaviate | gte_small | HYBRID_alpha0.75 | 31.6 | 31.666 | 31.669 | 32.196 | 32.273 | 0 |
| 328 | qdrant | gte_small | EXACT | 31.6 | 31.688 | 31.586 | 32.842 | 33.288 | 0 |
| 329 | qdrant | e5_small_hf | HNSW_ef32 | 31.6 | 31.691 | 31.551 | 32.614 | 32.745 | 0 |
| 330 | qdrant | paraphrase_multi | HNSW_default | 31.5 | 31.698 | 31.735 | 32.498 | 32.518 | 0 |
| 331 | qdrant | mpnet_multi | HNSW_ef256 | 31.5 | 31.713 | 31.688 | 32.433 | 32.537 | 0 |
| 332 | qdrant | bge_small_en | HNSW_default | 31.5 | 31.744 | 31.61 | 32.751 | 32.89 | 0 |
| 333 | weaviate | bge_base_en | HYBRID_alpha0.25 | 31.5 | 31.752 | 31.518 | 32.646 | 32.693 | 0 |
| 334 | qdrant | gte_small | HNSW_ef128 | 31.5 | 31.79 | 31.925 | 32.818 | 33.076 | 0 |
| 335 | weaviate | bge_base_en | HYBRID_alpha0.5 | 31.5 | 31.793 | 31.598 | 32.761 | 32.809 | 0 |
| 336 | qdrant | gte_small | HNSW_ef32 | 31.5 | 31.795 | 31.82 | 32.008 | 32.014 | 0 |
| 337 | qdrant | e5_base | HNSW_ef128 | 31.4 | 31.869 | 31.814 | 32.346 | 32.513 | 0 |
| 338 | qdrant | gte_small | HNSW_default | 31.4 | 31.875 | 31.84 | 32.641 | 33.021 | 0 |
| 339 | qdrant | all_mini_l6_base | EXACT | 31.4 | 31.891 | 31.9 | 32.121 | 32.163 | 0 |
| 340 | qdrant | bge_small_en | HNSW_ef32 | 31.3 | 31.906 | 31.825 | 33.021 | 33.387 | 0 |
| 341 | qdrant | multi_qa_minilm | HNSW_default | 31.3 | 31.968 | 31.941 | 32.811 | 32.928 | 0 |
| 342 | qdrant | e5_small | HNSW_ef128 | 31.3 | 31.982 | 32.055 | 33.193 | 33.292 | 0 |
| 343 | weaviate | distilroberta | HYBRID_alpha0.25 | 31.2 | 32.036 | 32.024 | 32.656 | 32.702 | 0 |
| 344 | qdrant | e5_base_base | HNSW_ef256 | 31.2 | 32.057 | 31.842 | 33.578 | 34.153 | 0 |
| 345 | weaviate | mpnet_multi_base | HNSW_limit50 | 31.1 | 32.149 | 32.031 | 33.22 | 33.264 | 0 |
| 346 | qdrant | bge_small_en | EXACT | 31.1 | 32.152 | 32.074 | 32.551 | 32.694 | 0 |
| 347 | qdrant | all_mini_l6 | HNSW_ef32 | 31 | 32.249 | 32.325 | 33.153 | 33.41 | 0 |
| 348 | qdrant | all_mini_l6_base | HNSW_ef256 | 31 | 32.274 | 32.072 | 33.372 | 33.961 | 0 |
| 349 | qdrant | all_mini_l6 | EXACT | 31 | 32.291 | 32.275 | 33.055 | 33.292 | 0 |
| 350 | qdrant | snowflake_arctic_l | HNSW_ef128 | 31 | 32.302 | 32.149 | 33.683 | 34.553 | 0 |
| 351 | qdrant | multi_qa_minilm | EXACT | 30.9 | 32.346 | 32.187 | 33.718 | 34.062 | 0 |
| 352 | qdrant | mpnet_multi_base | HNSW_ef256 | 30.9 | 32.355 | 32.374 | 32.557 | 32.642 | 0 |
| 353 | qdrant | minilm_l12 | HNSW_default | 30.9 | 32.357 | 32.419 | 33.16 | 33.39 | 0 |
| 354 | qdrant | all_mini_l6_base | HNSW_ef32 | 30.9 | 32.369 | 32.229 | 33.005 | 33.046 | 0 |
| 355 | qdrant | minilm_l12 | EXACT | 30.9 | 32.396 | 32.255 | 33.564 | 33.783 | 0 |
| 356 | qdrant | e5_small_hf | HNSW_default | 30.9 | 32.402 | 32.185 | 33.226 | 33.252 | 0 |
| 357 | weaviate | multi_qa_minilm | HYBRID_alpha0.25 | 30.9 | 32.409 | 32.055 | 34.227 | 34.305 | 0 |
| 358 | qdrant | multi_qa_minilm | HNSW_ef128 | 30.8 | 32.434 | 32.494 | 33.551 | 33.846 | 0 |
| 359 | qdrant | gte_small | HNSW_ef256 | 30.8 | 32.468 | 32.583 | 33.477 | 33.533 | 0 |
| 360 | qdrant | e5_small_base | EXACT | 30.8 | 32.478 | 32.104 | 33.809 | 33.838 | 0 |
| 361 | qdrant | all_mini_l6_base | HNSW_ef128 | 30.8 | 32.494 | 32.49 | 33.146 | 33.236 | 0 |
| 362 | qdrant | e5_small_hf | HNSW_ef128 | 30.7 | 32.537 | 32.557 | 33.133 | 33.213 | 0 |
| 363 | qdrant | gte_base | HNSW_ef256 | 30.7 | 32.549 | 32.742 | 33.409 | 33.496 | 0 |
| 364 | qdrant | e5_small_hf | EXACT | 30.7 | 32.552 | 32.541 | 33.284 | 33.679 | 0 |
| 365 | qdrant | e5_small_base | HNSW_ef32 | 30.7 | 32.597 | 31.931 | 35.647 | 37.91 | 0 |
| 366 | qdrant | all_mini_l6_base | HNSW_default | 30.7 | 32.597 | 32.569 | 33.141 | 33.151 | 0 |
| 367 | qdrant | multi_qa_minilm | HNSW_ef32 | 30.7 | 32.614 | 32.877 | 33.318 | 33.568 | 0 |
| 368 | qdrant | e5_small | HNSW_ef256 | 30.6 | 32.675 | 32.618 | 33.794 | 34.559 | 0 |
| 369 | qdrant | e5_small | EXACT | 30.6 | 32.728 | 32.744 | 33.403 | 33.623 | 0 |
| 370 | weaviate | gte_base | HNSW_limit50 | 30.5 | 32.736 | 32.434 | 34.449 | 35.621 | 0 |
| 371 | qdrant | all_mini_l6 | HNSW_ef128 | 30.5 | 32.785 | 32.636 | 35.514 | 37.331 | 0 |
| 372 | qdrant | e5_small_base | HNSW_ef128 | 30.5 | 32.789 | 32.754 | 34.791 | 36.229 | 0 |
| 373 | qdrant | e5_small_hf | HNSW_ef256 | 30.3 | 32.958 | 32.684 | 34.817 | 36.082 | 0 |
| 374 | qdrant | all_mini_l6 | HNSW_ef256 | 30.2 | 33.112 | 32.994 | 34.336 | 34.573 | 0 |
| 375 | qdrant | multi_qa_minilm | HNSW_ef256 | 30.2 | 33.123 | 33.015 | 33.613 | 33.683 | 0 |
| 376 | qdrant | multi_qa_mpnet | HNSW_ef256 | 30.1 | 33.254 | 33.217 | 33.632 | 33.742 | 0 |
| 377 | weaviate | bge_base_en | HYBRID_alpha0.75 | 30.1 | 33.264 | 33.112 | 34.531 | 34.727 | 0 |
| 378 | weaviate | multi_qa_mpnet | HNSW_limit50 | 30 | 33.308 | 33.295 | 33.762 | 33.767 | 0 |
| 379 | weaviate | mpnet_multi_base | HYBRID_alpha0.75 | 29.9 | 33.443 | 33.345 | 34.269 | 34.622 | 0 |
| 380 | weaviate | bge_squad_base | HNSW_limit50 | 29.9 | 33.473 | 33.443 | 34.232 | 34.364 | 0 |
| 381 | qdrant | e5_small | HNSW_default | 29.7 | 33.634 | 33.508 | 34.501 | 34.76 | 0 |
| 382 | weaviate | e5_small_base | HYBRID_alpha0.5 | 29.7 | 33.719 | 33.355 | 34.904 | 34.972 | 0 |
| 383 | qdrant | e5_small_base | HNSW_default | 29.4 | 33.974 | 33.694 | 36.775 | 38.39 | 0 |
| 384 | qdrant | e5_base_hf | HNSW_ef256 | 29.4 | 34.041 | 34.647 | 35.733 | 36.09 | 0 |
| 385 | weaviate | e5_small_hf | HYBRID_alpha0.25 | 29.1 | 34.409 | 34.429 | 34.976 | 35.08 | 0 |
| 386 | weaviate | e5_small_base | HYBRID_alpha0.25 | 28.9 | 34.543 | 33.688 | 36.798 | 36.887 | 0 |
| 387 | weaviate | gte_base | HYBRID_alpha0.75 | 28.9 | 34.585 | 34.604 | 34.879 | 34.934 | 0 |
| 388 | weaviate | e5_small_base | HYBRID_alpha0.75 | 28.8 | 34.742 | 34.233 | 36.98 | 37.207 | 0 |
| 389 | qdrant | bge_squad_base | HNSW_ef256 | 28.8 | 34.757 | 34.64 | 35.232 | 35.282 | 0 |
| 390 | weaviate | snowflake_arctic_l_base | HNSW_limit50 | 28.7 | 34.82 | 34.812 | 35.438 | 35.535 | 0 |
| 391 | weaviate | gte_base | HYBRID_alpha0.25 | 28.7 | 34.868 | 34.83 | 35.201 | 35.256 | 0 |
| 392 | weaviate | mpnet_multi | HNSW_limit50 | 28.4 | 35.248 | 35.48 | 36.298 | 36.379 | 0 |
| 393 | qdrant | snowflake_arctic_l_base | HNSW_ef256 | 28.3 | 35.355 | 35.091 | 36.688 | 37.194 | 0 |
| 394 | weaviate | all_mini_l6 | HNSW_limit50 | 28.2 | 35.507 | 35.311 | 36.794 | 37.702 | 0 |
| 395 | weaviate | gte_base | HYBRID_alpha0.5 | 28.1 | 35.559 | 35.601 | 36.532 | 36.579 | 0 |
| 396 | weaviate | mpnet_multi_base | HYBRID_alpha0.5 | 28.1 | 35.626 | 36.016 | 36.895 | 37.063 | 0 |
| 397 | weaviate | e5_base_hf | HNSW_limit50 | 27.9 | 35.891 | 35.86 | 36.587 | 36.695 | 0 |
| 398 | weaviate | e5_base_base | HNSW_limit50 | 27.8 | 35.917 | 35.424 | 39.267 | 41.272 | 0 |
| 399 | weaviate | multi_qa_mpnet | HYBRID_alpha0.75 | 27.6 | 36.22 | 36.392 | 36.963 | 37.194 | 0 |
| 400 | weaviate | multi_qa_mpnet | HYBRID_alpha0.5 | 27.5 | 36.421 | 36.464 | 37.642 | 37.676 | 0 |
| 401 | weaviate | e5_small_hf | HYBRID_alpha0.75 | 27.4 | 36.433 | 36.528 | 37.31 | 37.359 | 0 |
| 402 | qdrant | bge_squad | HNSW_ef256 | 27.4 | 36.507 | 36.278 | 37.766 | 37.868 | 0 |
| 403 | weaviate | e5_small | HNSW_limit50 | 27 | 37.064 | 36.794 | 38.11 | 38.149 | 0 |
| 404 | weaviate | multi_qa_mpnet | HYBRID_alpha0.25 | 26.9 | 37.118 | 36.885 | 38.358 | 38.742 | 0 |
| 405 | weaviate | bge_squad | HNSW_limit50 | 26.9 | 37.128 | 37.009 | 38.663 | 39.042 | 0 |
| 406 | weaviate | bge_squad_base | HYBRID_alpha0.5 | 26.5 | 37.673 | 37.924 | 38.705 | 39.143 | 0 |
| 407 | weaviate | e5_small_hf | HYBRID_alpha0.5 | 25.9 | 38.625 | 34.391 | 55.74 | 69.456 | 0 |
| 408 | weaviate | bge_squad_base | HYBRID_alpha0.75 | 25.8 | 38.76 | 38.363 | 41.252 | 42.235 | 0 |
| 409 | weaviate | bge_squad_base | HYBRID_alpha0.25 | 25.7 | 38.87 | 39.041 | 39.666 | 39.909 | 0 |
| 410 | weaviate | snowflake_arctic_l_base | HYBRID_alpha0.75 | 25.7 | 38.924 | 38.632 | 40.628 | 41.172 | 0 |
| 411 | weaviate | e5_base_base | HYBRID_alpha0.75 | 25.6 | 39.047 | 39.101 | 40.059 | 40.094 | 0 |
| 412 | weaviate | snowflake_arctic_l | HNSW_limit50 | 25.4 | 39.363 | 39.289 | 40.799 | 41.126 | 0 |
| 413 | weaviate | snowflake_arctic_l_base | HYBRID_alpha0.5 | 25.4 | 39.412 | 39.742 | 40.402 | 40.462 | 0 |
| 414 | weaviate | snowflake_arctic_l_base | HYBRID_alpha0.25 | 25 | 39.938 | 40.041 | 40.821 | 40.827 | 0 |
| 415 | weaviate | mpnet_multi | HYBRID_alpha0.5 | 25 | 39.983 | 39.098 | 43.72 | 45.227 | 0 |
| 416 | weaviate | all_mini_l6 | HYBRID_alpha0.75 | 25 | 40.029 | 39.745 | 42.517 | 43.741 | 0 |
| 417 | weaviate | all_mini_l6 | HYBRID_alpha0.25 | 24.9 | 40.132 | 40.167 | 41.292 | 41.321 | 0 |
| 418 | weaviate | mpnet_multi | HYBRID_alpha0.75 | 24.9 | 40.167 | 40.03 | 42.1 | 42.69 | 0 |
| 419 | weaviate | mpnet_multi | HYBRID_alpha0.25 | 24.8 | 40.252 | 39.046 | 44.359 | 44.777 | 0 |
| 420 | weaviate | e5_base_hf | HYBRID_alpha0.25 | 24.8 | 40.288 | 40.649 | 41.496 | 41.511 | 0 |
| 421 | qdrant | e5_base | HNSW_ef256 | 24.7 | 40.408 | 40.633 | 41.975 | 42.114 | 0 |
| 422 | weaviate | e5_base_base | HYBRID_alpha0.5 | 24.5 | 40.882 | 40.818 | 42.163 | 42.832 | 0 |
| 423 | weaviate | mpnet_multi_base | HYBRID_alpha0.25 | 24.5 | 40.892 | 37.571 | 59.494 | 69.254 | 0 |
| 424 | weaviate | e5_base_hf | HYBRID_alpha0.5 | 24.4 | 40.962 | 40.968 | 41.6 | 41.695 | 0 |
| 425 | weaviate | e5_base_hf | HYBRID_alpha0.75 | 24.4 | 40.995 | 40.923 | 42.117 | 42.225 | 0 |
| 426 | qdrant | snowflake_arctic_l | HNSW_ef256 | 24.3 | 41.072 | 41.191 | 42.42 | 42.756 | 0 |
| 427 | weaviate | e5_base_base | HYBRID_alpha0.25 | 24.1 | 41.47 | 41.73 | 42.758 | 43.149 | 0 |
| 428 | weaviate | e5_small | HYBRID_alpha0.25 | 24 | 41.71 | 41.861 | 42.759 | 43.289 | 0 |
| 429 | weaviate | bge_squad | HYBRID_alpha0.25 | 23.7 | 42.267 | 42.229 | 43.551 | 43.745 | 0 |
| 430 | weaviate | bge_squad | HYBRID_alpha0.75 | 23.5 | 42.6 | 42.876 | 44.383 | 44.407 | 0 |
| 431 | weaviate | bge_squad | HYBRID_alpha0.5 | 23.5 | 42.629 | 42.742 | 44.204 | 44.503 | 0 |
| 432 | weaviate | all_mini_l6 | HYBRID_alpha0.5 | 22.9 | 43.739 | 40.094 | 59.513 | 73.929 | 0 |
| 433 | weaviate | e5_base | HNSW_limit50 | 22.7 | 44.04 | 43.755 | 46.635 | 47.065 | 0 |
| 434 | weaviate | snowflake_arctic_l | HYBRID_alpha0.5 | 21.6 | 46.375 | 46.288 | 48.303 | 48.316 | 0 |
| 435 | weaviate | snowflake_arctic_l | HYBRID_alpha0.25 | 21.6 | 46.394 | 46.476 | 47.578 | 47.702 | 0 |
| 436 | weaviate | snowflake_arctic_l | HYBRID_alpha0.75 | 21.4 | 46.721 | 46.621 | 47.633 | 47.683 | 0 |
| 437 | qdrant | e5_base_hf | HNSW_default | 20.2 | 49.591 | 27.395 | 141.105 | 165.641 | 0 |
| 438 | weaviate | e5_base | HYBRID_alpha0.25 | 20.2 | 49.605 | 49.548 | 50.846 | 51.089 | 0 |
| 439 | weaviate | e5_base | HYBRID_alpha0.75 | 20 | 49.934 | 49.923 | 53.259 | 55.537 | 0 |
| 440 | weaviate | e5_base | HYBRID_alpha0.5 | 19.9 | 50.314 | 50.149 | 52.634 | 54.272 | 0 |
| 441 | weaviate | e5_small | HYBRID_alpha0.5 | 19.7 | 50.823 | 44.547 | 79.038 | 93.092 | 0 |
| 442 | qdrant | e5_base | EXACT | 18 | 55.686 | 55.67 | 57.156 | 57.694 | 0 |
| 443 | qdrant | mpnet_multi_base | EXACT | 17.8 | 56.143 | 55.805 | 57.419 | 57.606 | 0 |
| 444 | qdrant | e5_base_hf | EXACT | 17.7 | 56.352 | 56.279 | 57.686 | 57.719 | 0 |
| 445 | qdrant | mpnet_multi | EXACT | 17.7 | 56.456 | 56.081 | 59.837 | 60.754 | 0 |
| 446 | qdrant | e5_base_base | EXACT | 17.3 | 57.747 | 57.867 | 59.837 | 59.98 | 0 |
| 447 | qdrant | distilroberta | EXACT | 17.3 | 57.878 | 57.843 | 61.309 | 63.529 | 0 |
| 448 | qdrant | bge_base_en | EXACT | 17.3 | 57.943 | 57.77 | 59.574 | 60.154 | 0 |
| 449 | qdrant | mpnet_base | EXACT | 17.3 | 57.963 | 57.917 | 59.547 | 59.554 | 0 |
| 450 | qdrant | multi_qa_mpnet | EXACT | 17.2 | 57.99 | 57.764 | 59.37 | 59.508 | 0 |
| 451 | qdrant | gte_base | EXACT | 17 | 58.965 | 58.667 | 62.244 | 63.967 | 0 |
| 452 | weaviate | e5_small | HYBRID_alpha0.75 | 16.7 | 59.726 | 47.118 | 89.715 | 90.063 | 0 |
| 453 | qdrant | snowflake_arctic_l_base | EXACT | 14.3 | 69.713 | 69.596 | 71.148 | 71.533 | 0 |
| 454 | qdrant | bge_squad | EXACT | 13.6 | 73.479 | 73.933 | 75.746 | 75.751 | 0 |
| 455 | qdrant | bge_squad_base | EXACT | 13.6 | 73.632 | 73.443 | 76.198 | 76.404 | 0 |
| 456 | qdrant | snowflake_arctic_l | EXACT | 13.4 | 74.538 | 74.879 | 76.826 | 77.361 | 0 |
| 457 | milvus | bge_small_en | HNSW_limit20 | 13.3 | 75.179 | 74.964 | 77.086 | 77.347 | 0 |
| 458 | milvus | paraphrase_multi | HNSW_limit20 | 13.1 | 76.441 | 76.195 | 79.01 | 80.118 | 0 |
| 459 | milvus | bge_small_en | HNSW_limit50 | 13 | 76.886 | 76.753 | 78.732 | 79.404 | 0 |
| 460 | milvus | paraphrase_multi | HNSW_limit5 | 13 | 77.187 | 77.455 | 78.225 | 78.39 | 0 |
| 461 | milvus | bge_small_en | HNSW_limit5 | 12.9 | 77.276 | 76.795 | 82.539 | 85.302 | 0 |
| 462 | milvus | all_mini_l6 | HNSW_limit5 | 12.9 | 77.302 | 77.364 | 78.408 | 78.545 | 0 |
| 463 | milvus | paraphrase_multi | HNSW_default | 12.9 | 77.33 | 77.131 | 78.204 | 78.207 | 0 |
| 464 | milvus | gte_small | HNSW_default | 12.9 | 77.519 | 77.65 | 78.585 | 78.591 | 0 |
| 465 | milvus | e5_small_hf | HNSW_default | 12.9 | 77.689 | 77.51 | 80.339 | 80.815 | 0 |
| 466 | milvus | all_mini_l6_base | HNSW_limit20 | 12.9 | 77.704 | 77.595 | 78.342 | 78.355 | 0 |
| 467 | milvus | e5_small_base | HNSW_default | 12.9 | 77.734 | 77.736 | 78.217 | 78.24 | 0 |
| 468 | milvus | paraphrase_multi | HNSW_limit50 | 12.8 | 77.87 | 77.61 | 79.713 | 79.987 | 0 |
| 469 | milvus | minilm_l12 | HNSW_limit20 | 12.8 | 78.091 | 78.097 | 78.828 | 78.979 | 0 |
| 470 | milvus | e5_small_base | HNSW_limit5 | 12.8 | 78.093 | 78.096 | 79.046 | 79.232 | 0 |
| 471 | milvus | gte_small | HNSW_limit20 | 12.8 | 78.116 | 77.747 | 79.601 | 79.981 | 0 |
| 472 | milvus | all_mini_l6_base | HNSW_default | 12.8 | 78.183 | 78.3 | 79.356 | 79.402 | 0 |
| 473 | milvus | all_mini_l6 | HNSW_default | 12.8 | 78.274 | 78.252 | 78.719 | 78.868 | 0 |
| 474 | milvus | e5_small_hf | HNSW_limit5 | 12.7 | 78.445 | 78.55 | 79.596 | 79.729 | 0 |
| 475 | milvus | e5_small_hf | HNSW_limit20 | 12.7 | 78.462 | 78.294 | 79.513 | 79.526 | 0 |
| 476 | milvus | all_mini_l6 | HNSW_limit20 | 12.7 | 78.615 | 78.655 | 79.312 | 79.359 | 0 |
| 477 | milvus | gte_small | HNSW_limit50 | 12.7 | 78.772 | 78.68 | 80.115 | 80.779 | 0 |
| 478 | milvus | multi_qa_minilm | HNSW_limit20 | 12.7 | 78.782 | 76.684 | 86.25 | 87.592 | 0 |
| 479 | milvus | e5_small | HNSW_limit5 | 12.7 | 78.843 | 78.563 | 79.633 | 79.796 | 0 |
| 480 | milvus | all_mini_l6 | HNSW_limit50 | 12.7 | 78.927 | 78.878 | 79.69 | 80.04 | 0 |
| 481 | milvus | gte_small | HNSW_limit5 | 12.6 | 79.105 | 77.991 | 84.498 | 84.653 | 0 |
| 482 | milvus | minilm_l12 | HNSW_limit5 | 12.6 | 79.176 | 77.212 | 85.883 | 87.481 | 0 |
| 483 | milvus | bge_small_en | HNSW_default | 12.6 | 79.196 | 78.541 | 84.303 | 87.096 | 0 |
| 484 | milvus | minilm_l12 | HNSW_default | 12.6 | 79.236 | 78.892 | 80.757 | 80.858 | 0 |
| 485 | milvus | e5_small | HNSW_limit20 | 12.6 | 79.278 | 79.111 | 80.697 | 80.996 | 0 |
| 486 | milvus | multi_qa_minilm | HNSW_limit5 | 12.6 | 79.334 | 78.565 | 84.363 | 86.174 | 0 |
| 487 | milvus | multi_qa_minilm | HNSW_default | 12.6 | 79.338 | 78.699 | 82.591 | 83.029 | 0 |
| 488 | milvus | all_mini_l6_base | HNSW_limit50 | 12.6 | 79.673 | 79.695 | 80.373 | 80.461 | 0 |
| 489 | milvus | e5_small_hf | HNSW_limit50 | 12.5 | 79.833 | 79.453 | 82.157 | 83.363 | 0 |
| 490 | milvus | all_mini_l6_base | HNSW_limit5 | 12.5 | 79.883 | 79.899 | 81.656 | 82.118 | 0 |
| 491 | milvus | multi_qa_minilm | HNSW_limit50 | 12.5 | 80.211 | 80.14 | 80.921 | 80.981 | 0 |
| 492 | milvus | e5_small | HNSW_limit50 | 12.4 | 80.596 | 80.082 | 82.494 | 83.08 | 0 |
| 493 | milvus | minilm_l12 | HNSW_limit50 | 12.4 | 80.743 | 81.094 | 82.334 | 82.467 | 0 |
| 494 | milvus | e5_small | HNSW_default | 12.4 | 80.814 | 79.454 | 88.183 | 91.886 | 0 |
| 495 | milvus | e5_small_base | HNSW_limit50 | 12.3 | 81.179 | 80.202 | 85.767 | 86.623 | 0 |
| 496 | milvus | e5_small_base | HNSW_limit20 | 11.9 | 83.749 | 79.653 | 101.066 | 107.386 | 0 |
| 497 | milvus | bge_base_en | HNSW_limit20 | 7.9 | 126.964 | 127.134 | 127.784 | 127.928 | 0 |
| 498 | milvus | bge_base_en | HNSW_limit5 | 7.8 | 127.629 | 127.569 | 128.906 | 129.278 | 0 |
| 499 | milvus | mpnet_multi_base | HNSW_default | 7.8 | 128.168 | 128.031 | 129.276 | 129.388 | 0 |
| 500 | milvus | mpnet_multi_base | HNSW_limit5 | 7.8 | 128.227 | 128.077 | 130.949 | 132.398 | 0 |
| 501 | milvus | bge_base_en | HNSW_limit50 | 7.8 | 128.469 | 128.488 | 129.197 | 129.28 | 0 |
| 502 | milvus | e5_base | HNSW_limit5 | 7.8 | 128.59 | 128.52 | 129.747 | 129.818 | 0 |
| 503 | milvus | e5_base | HNSW_limit20 | 7.8 | 128.783 | 128.355 | 131.164 | 131.343 | 0 |
| 504 | milvus | e5_base | HNSW_default | 7.7 | 129.113 | 129.232 | 130.187 | 130.211 | 0 |
| 505 | milvus | mpnet_multi_base | HNSW_limit20 | 7.7 | 129.123 | 129.227 | 130.841 | 130.945 | 0 |
| 506 | milvus | e5_base_base | HNSW_limit5 | 7.7 | 129.156 | 129.856 | 131.649 | 132.541 | 0 |
| 507 | milvus | gte_base | HNSW_limit5 | 7.7 | 129.171 | 129.033 | 130.005 | 130.337 | 0 |
| 508 | milvus | bge_base_en | HNSW_default | 7.7 | 129.182 | 128.598 | 132.819 | 135.14 | 0 |
| 509 | milvus | e5_base_base | HNSW_default | 7.7 | 129.279 | 128.922 | 130.932 | 131.24 | 0 |
| 510 | milvus | mpnet_multi_base | HNSW_limit50 | 7.7 | 129.365 | 129.148 | 131.292 | 131.902 | 0 |
| 511 | milvus | gte_base | HNSW_default | 7.7 | 129.401 | 129.407 | 130.327 | 130.61 | 0 |
| 512 | milvus | gte_base | HNSW_limit20 | 7.7 | 129.624 | 129.514 | 130.976 | 131.378 | 0 |
| 513 | milvus | multi_qa_mpnet | HNSW_limit5 | 7.7 | 129.715 | 129.622 | 130.643 | 130.851 | 0 |
| 514 | milvus | multi_qa_mpnet | HNSW_limit20 | 7.7 | 129.832 | 129.837 | 130.587 | 130.838 | 0 |
| 515 | milvus | multi_qa_mpnet | HNSW_default | 7.7 | 130.164 | 130.097 | 131.096 | 131.296 | 0 |
| 516 | milvus | mpnet_multi | HNSW_limit5 | 7.7 | 130.268 | 129.919 | 132.35 | 132.731 | 0 |
| 517 | milvus | e5_base | HNSW_limit50 | 7.7 | 130.473 | 130.491 | 131.775 | 131.789 | 0 |
| 518 | milvus | e5_base_base | HNSW_limit20 | 7.6 | 130.784 | 130.696 | 131.937 | 132.092 | 0 |
| 519 | milvus | distilroberta | HNSW_limit20 | 7.6 | 130.862 | 130.946 | 131.958 | 131.966 | 0 |
| 520 | milvus | mpnet_base | HNSW_default | 7.6 | 131.031 | 131.143 | 131.931 | 132.161 | 0 |
| 521 | milvus | multi_qa_mpnet | HNSW_limit50 | 7.6 | 131.178 | 130.958 | 132.92 | 133.024 | 0 |
| 522 | milvus | e5_base_hf | HNSW_limit5 | 7.6 | 131.252 | 130.555 | 135.591 | 137.868 | 0 |
| 523 | milvus | e5_base_hf | HNSW_default | 7.6 | 131.398 | 131.322 | 132.393 | 132.468 | 0 |
| 524 | milvus | distilroberta | HNSW_default | 7.6 | 131.517 | 131.338 | 133.429 | 133.646 | 0 |
| 525 | milvus | mpnet_multi | HNSW_limit50 | 7.6 | 131.685 | 131.553 | 132.824 | 133.097 | 0 |
| 526 | milvus | gte_base | HNSW_limit50 | 7.6 | 131.728 | 131.681 | 133.767 | 134.221 | 0 |
| 527 | milvus | mpnet_multi | HNSW_default | 7.6 | 131.752 | 131.502 | 133.33 | 133.379 | 0 |
| 528 | milvus | distilroberta | HNSW_limit5 | 7.6 | 131.764 | 131.912 | 133.052 | 133.194 | 0 |
| 529 | milvus | mpnet_base | HNSW_limit5 | 7.6 | 131.981 | 130.785 | 138.336 | 143.527 | 0 |
| 530 | milvus | e5_base_hf | HNSW_limit20 | 7.6 | 132.236 | 131.787 | 134.591 | 135.018 | 0 |
| 531 | milvus | distilroberta | HNSW_limit50 | 7.6 | 132.383 | 132.161 | 133.634 | 133.711 | 0 |
| 532 | milvus | e5_base_base | HNSW_limit50 | 7.6 | 132.39 | 132.012 | 133.873 | 134.496 | 0 |
| 533 | milvus | e5_base_hf | HNSW_limit50 | 7.5 | 132.462 | 132.066 | 135.217 | 135.764 | 0 |
| 534 | milvus | mpnet_base | HNSW_limit50 | 7.5 | 132.601 | 133.02 | 134.098 | 134.11 | 0 |
| 535 | milvus | mpnet_multi | HNSW_limit20 | 7.5 | 133.339 | 132.946 | 137.178 | 139.155 | 0 |
| 536 | milvus | mpnet_base | HNSW_limit20 | 7.5 | 134.216 | 132.447 | 141.717 | 143.251 | 0 |
| 537 | milvus | bge_squad_base | HNSW_limit5 | 6.3 | 159.339 | 159.649 | 161.116 | 161.339 | 0 |
| 538 | milvus | bge_squad_base | HNSW_limit20 | 6.2 | 160.19 | 159.904 | 161.818 | 162.289 | 0 |
| 539 | milvus | bge_squad_base | HNSW_default | 6.2 | 160.2 | 159.95 | 163.406 | 163.826 | 0 |
| 540 | milvus | bge_squad_base | HNSW_limit50 | 6.2 | 160.91 | 161.036 | 162.461 | 162.835 | 0 |
| 541 | milvus | snowflake_arctic_l_base | HNSW_limit5 | 6.2 | 161.148 | 161.133 | 162.354 | 162.916 | 0 |
| 542 | milvus | snowflake_arctic_l_base | HNSW_limit20 | 6.2 | 162.093 | 161.693 | 164.242 | 164.832 | 0 |
| 543 | milvus | snowflake_arctic_l_base | HNSW_default | 6.2 | 162.523 | 162.221 | 164.256 | 164.335 | 0 |
| 544 | milvus | snowflake_arctic_l_base | HNSW_limit50 | 6.1 | 164.768 | 164.39 | 166.764 | 167.475 | 0 |
| 545 | milvus | bge_squad | HNSW_default | 6 | 166.793 | 166.417 | 167.998 | 168.308 | 0 |
| 546 | milvus | bge_squad | HNSW_limit50 | 6 | 167.385 | 167.355 | 168.43 | 168.452 | 0 |
| 547 | milvus | bge_squad | HNSW_limit5 | 6 | 167.763 | 167.332 | 172.533 | 175.393 | 0 |
| 548 | milvus | bge_squad | HNSW_limit20 | 5.9 | 168.097 | 167.581 | 171.955 | 173.688 | 0 |
| 549 | milvus | snowflake_arctic_l | HNSW_limit5 | 5.9 | 170.208 | 169.861 | 171.78 | 171.838 | 0 |
| 550 | milvus | snowflake_arctic_l | HNSW_limit20 | 5.9 | 170.418 | 170.389 | 171.586 | 171.901 | 0 |
| 551 | milvus | snowflake_arctic_l | HNSW_limit50 | 5.8 | 171.06 | 171.442 | 172.59 | 172.905 | 0 |
| 552 | milvus | snowflake_arctic_l | HNSW_default | 5.7 | 173.956 | 171.418 | 185.051 | 187.675 | 0 |
| 553 | lancedb | e5_small | VECTOR_default | 2.2 | 451.783 | 451.928 | 456.22 | 457.505 | 0 |
| 554 | lancedb | e5_small | VECTOR_limit5 | 2.2 | 458.725 | 453.762 | 486.303 | 504.818 | 0 |
| 555 | lancedb | e5_small | VECTOR_cosine | 2.2 | 460.358 | 460.227 | 466.726 | 469.308 | 0 |
| 556 | lancedb | multi_qa_minilm | VECTOR_default | 2.2 | 461.814 | 462.338 | 468.237 | 471.286 | 0 |
| 557 | lancedb | multi_qa_minilm | VECTOR_cosine | 2.2 | 464.111 | 463.824 | 469.145 | 470.921 | 0 |
| 558 | lancedb | all_mini_l6_base | VECTOR_default | 2.2 | 464.431 | 464.057 | 471.2 | 472.446 | 0 |
| 559 | lancedb | all_mini_l6_base | VECTOR_limit5 | 2.2 | 464.508 | 459.828 | 488.723 | 510.767 | 0 |
| 560 | lancedb | multi_qa_minilm | VECTOR_limit5 | 2.1 | 465.176 | 460.772 | 490.875 | 511.363 | 0 |
| 561 | lancedb | all_mini_l6_base | VECTOR_cosine | 2.1 | 465.835 | 465.03 | 469.927 | 470.379 | 0 |
| 562 | lancedb | all_mini_l6_base | VECTOR_L2 | 2.1 | 465.848 | 465.514 | 470.723 | 470.801 | 0 |
| 563 | lancedb | gte_small | VECTOR_cosine | 2.1 | 466.035 | 466.156 | 470.229 | 471.198 | 0 |
| 564 | lancedb | gte_small | VECTOR_default | 2.1 | 467.021 | 466.616 | 473.71 | 476.42 | 0 |
| 565 | lancedb | gte_small | VECTOR_L2 | 2.1 | 467.224 | 466.177 | 475.165 | 476.115 | 0 |
| 566 | lancedb | e5_small | VECTOR_limit20 | 2.1 | 467.504 | 466.937 | 471.57 | 472.329 | 0 |
| 567 | lancedb | multi_qa_minilm | VECTOR_L2 | 2.1 | 467.999 | 466.714 | 472.225 | 473.464 | 0 |
| 568 | lancedb | all_mini_l6 | VECTOR_default | 2.1 | 468.509 | 467.934 | 473.852 | 474.968 | 0 |
| 569 | lancedb | gte_small | VECTOR_limit5 | 2.1 | 468.712 | 465.142 | 493.494 | 514.023 | 0 |
| 570 | lancedb | e5_small_hf | VECTOR_default | 2.1 | 469.366 | 468.96 | 477.416 | 478.909 | 0 |
| 571 | lancedb | e5_small_hf | VECTOR_cosine | 2.1 | 469.577 | 469.718 | 473.659 | 473.775 | 0 |
| 572 | lancedb | e5_small | VECTOR_L2 | 2.1 | 469.833 | 466.645 | 490.796 | 492.823 | 0 |
| 573 | lancedb | paraphrase_multi | VECTOR_limit5 | 2.1 | 469.867 | 465.914 | 493.627 | 514.091 | 0 |
| 574 | lancedb | e5_small_base | VECTOR_limit5 | 2.1 | 470.086 | 465.31 | 494.882 | 512.526 | 0 |
| 575 | lancedb | paraphrase_multi | VECTOR_cosine | 2.1 | 470.295 | 470.502 | 473.122 | 473.471 | 0 |
| 576 | lancedb | paraphrase_multi | VECTOR_default | 2.1 | 471.66 | 467.652 | 493.079 | 510.462 | 0 |
| 577 | lancedb | bge_small_en | VECTOR_limit5 | 2.1 | 471.662 | 466.251 | 498.888 | 523.103 | 0 |
| 578 | lancedb | all_mini_l6_base | VECTOR_limit20 | 2.1 | 472.198 | 471.48 | 479.579 | 484.15 | 0 |
| 579 | lancedb | e5_small_base | VECTOR_default | 2.1 | 472.292 | 470.672 | 481.911 | 484.89 | 0 |
| 580 | lancedb | bge_small_en | VECTOR_default | 2.1 | 472.593 | 472.844 | 477.303 | 477.328 | 0 |
| 581 | lancedb | bge_small_en | VECTOR_cosine | 2.1 | 472.707 | 472.116 | 480.068 | 480.418 | 0 |
| 582 | lancedb | all_mini_l6 | VECTOR_cosine | 2.1 | 472.735 | 472.832 | 476.358 | 476.78 | 0 |
| 583 | lancedb | minilm_l12 | VECTOR_L2 | 2.1 | 473.026 | 470.566 | 483.69 | 485.156 | 0 |
| 584 | lancedb | minilm_l12 | VECTOR_cosine | 2.1 | 473.038 | 472.408 | 477.545 | 478.238 | 0 |
| 585 | lancedb | e5_small_base | VECTOR_cosine | 2.1 | 473.205 | 472.183 | 479.797 | 481.055 | 0 |
| 586 | lancedb | e5_small_hf | VECTOR_limit5 | 2.1 | 473.292 | 469.224 | 500.062 | 522.549 | 0 |
| 587 | lancedb | e5_small_base | VECTOR_L2 | 2.1 | 473.446 | 473.563 | 479.092 | 479.414 | 0 |
| 588 | lancedb | e5_small_hf | VECTOR_L2 | 2.1 | 473.68 | 473.286 | 479.163 | 481.117 | 0 |
| 589 | lancedb | paraphrase_multi | VECTOR_L2 | 2.1 | 473.91 | 472.61 | 487.428 | 491.216 | 0 |
| 590 | lancedb | minilm_l12 | VECTOR_default | 2.1 | 474.029 | 474.066 | 480.265 | 482.933 | 0 |
| 591 | lancedb | all_mini_l6 | VECTOR_L2 | 2.1 | 474.04 | 472.85 | 489.234 | 495.768 | 0 |
| 592 | lancedb | multi_qa_minilm | VECTOR_limit20 | 2.1 | 474.162 | 472.588 | 485.641 | 492.217 | 0 |
| 593 | lancedb | gte_base | VECTOR_default | 2.1 | 474.25 | 473.186 | 481.833 | 485.756 | 0 |
| 594 | lancedb | all_mini_l6 | VECTOR_limit5 | 2.1 | 474.256 | 467.259 | 502.572 | 525.457 | 0 |
| 595 | lancedb | e5_base | VECTOR_default | 2.1 | 474.639 | 473.902 | 479.218 | 480.265 | 0 |
| 596 | lancedb | minilm_l12 | VECTOR_limit5 | 2.1 | 474.802 | 469.212 | 503.945 | 526.77 | 0 |
| 597 | lancedb | paraphrase_multi | VECTOR_limit20 | 2.1 | 475.51 | 476.603 | 477.923 | 478.179 | 0 |
| 598 | lancedb | bge_small_en | VECTOR_L2 | 2.1 | 476.032 | 473.812 | 485.595 | 487.054 | 0 |
| 599 | lancedb | gte_small | VECTOR_limit20 | 2.1 | 476.063 | 474.018 | 488.629 | 495.699 | 0 |
| 600 | lancedb | gte_base | VECTOR_limit5 | 2.1 | 476.169 | 470.079 | 505.862 | 531.087 | 0 |
| 601 | lancedb | mpnet_multi_base | VECTOR_default | 2.1 | 476.374 | 475.592 | 480.832 | 481.819 | 0 |
| 602 | lancedb | gte_base | VECTOR_cosine | 2.1 | 476.406 | 476.51 | 477.728 | 477.87 | 0 |
| 603 | lancedb | e5_base | VECTOR_limit5 | 2.1 | 476.955 | 471.819 | 501.182 | 523.335 | 0 |
| 604 | lancedb | mpnet_multi | VECTOR_default | 2.1 | 477.063 | 474.976 | 484.807 | 485.322 | 0 |
| 605 | lancedb | mpnet_multi_base | VECTOR_limit5 | 2.1 | 477.175 | 473.033 | 499.529 | 520.413 | 0 |
| 606 | lancedb | bge_small_en | VECTOR_limit20 | 2.1 | 477.414 | 476.902 | 482.559 | 483.078 | 0 |
| 607 | lancedb | all_mini_l6 | VECTOR_limit20 | 2.1 | 477.781 | 476.799 | 484.575 | 488.626 | 0 |
| 608 | lancedb | mpnet_multi_base | VECTOR_cosine | 2.1 | 478.136 | 477.672 | 481.787 | 484.155 | 0 |
| 609 | lancedb | gte_base | VECTOR_L2 | 2.1 | 478.343 | 477.525 | 485.184 | 487.831 | 0 |
| 610 | lancedb | mpnet_multi | VECTOR_limit5 | 2.1 | 478.653 | 473.298 | 503.169 | 524.587 | 0 |
| 611 | lancedb | mpnet_base | VECTOR_default | 2.1 | 478.941 | 477.337 | 485.342 | 490.479 | 0 |
| 612 | lancedb | distilroberta | VECTOR_L2 | 2.1 | 479.217 | 479.18 | 482.229 | 482.809 | 0 |
| 613 | lancedb | minilm_l12 | VECTOR_limit20 | 2.1 | 479.326 | 479.383 | 486.037 | 487.972 | 0 |
| 614 | lancedb | mpnet_multi_base | VECTOR_L2 | 2.1 | 479.363 | 478.875 | 481.916 | 482.201 | 0 |
| 615 | lancedb | mpnet_base | VECTOR_limit5 | 2.1 | 479.465 | 474.222 | 507.118 | 529.338 | 0 |
| 616 | lancedb | e5_small_base | VECTOR_limit20 | 2.1 | 479.61 | 481.01 | 482.188 | 482.781 | 0 |
| 617 | lancedb | e5_small_hf | VECTOR_limit20 | 2.1 | 479.833 | 479.946 | 485.45 | 485.909 | 0 |
| 618 | lancedb | mpnet_multi | VECTOR_cosine | 2.1 | 480.064 | 480.091 | 484.739 | 484.794 | 0 |
| 619 | lancedb | mpnet_base | VECTOR_cosine | 2.1 | 480.241 | 480.443 | 485.514 | 486.062 | 0 |
| 620 | lancedb | e5_base_hf | VECTOR_default | 2.1 | 480.295 | 478.927 | 489.04 | 493.837 | 0 |
| 621 | lancedb | distilroberta | VECTOR_cosine | 2.1 | 480.3 | 480.872 | 483.546 | 483.82 | 0 |
| 622 | lancedb | mpnet_multi | VECTOR_L2 | 2.1 | 480.322 | 480.32 | 483.485 | 483.714 | 0 |
| 623 | lancedb | e5_base_hf | VECTOR_limit5 | 2.1 | 480.404 | 475.752 | 506.855 | 528.378 | 0 |
| 624 | lancedb | e5_base_hf | VECTOR_L2 | 2.1 | 480.523 | 480.643 | 484.5 | 485.912 | 0 |
| 625 | lancedb | e5_base_hf | VECTOR_cosine | 2.1 | 480.528 | 480.883 | 482.587 | 483.046 | 0 |
| 626 | lancedb | distilroberta | VECTOR_limit5 | 2.1 | 480.87 | 475.418 | 505.67 | 527.048 | 0 |
| 627 | lancedb | gte_base | VECTOR_limit20 | 2.1 | 480.926 | 481.266 | 484.367 | 484.938 | 0 |
| 628 | lancedb | multi_qa_mpnet | VECTOR_L2 | 2.1 | 481.107 | 481.796 | 484.817 | 485.152 | 0 |
| 629 | lancedb | multi_qa_mpnet | VECTOR_default | 2.1 | 481.138 | 481.58 | 486.784 | 488.929 | 0 |
| 630 | lancedb | distilroberta | VECTOR_default | 2.1 | 481.168 | 479.623 | 490.7 | 492.598 | 0 |
| 631 | lancedb | multi_qa_mpnet | VECTOR_limit5 | 2.1 | 481.254 | 476.996 | 504.429 | 523.326 | 0 |
| 632 | lancedb | bge_base_en | VECTOR_default | 2.1 | 481.258 | 481.401 | 485.226 | 486.034 | 0 |
| 633 | lancedb | multi_qa_mpnet | VECTOR_cosine | 2.1 | 481.295 | 481.023 | 486.193 | 486.86 | 0 |
| 634 | lancedb | bge_base_en | VECTOR_limit5 | 2.1 | 481.319 | 477.786 | 505.668 | 526.086 | 0 |
| 635 | lancedb | snowflake_arctic_l | VECTOR_limit5 | 2.1 | 481.695 | 477.855 | 505.26 | 526.56 | 0 |
| 636 | lancedb | snowflake_arctic_l_base | VECTOR_limit5 | 2.1 | 481.964 | 477.944 | 504.669 | 524.382 | 0 |
| 637 | lancedb | e5_base_base | VECTOR_limit5 | 2.1 | 482.094 | 476.58 | 506.208 | 526.392 | 0 |
| 638 | lancedb | e5_base_base | VECTOR_default | 2.1 | 482.172 | 480.476 | 489.453 | 490.85 | 0 |
| 639 | lancedb | e5_base_base | VECTOR_L2 | 2.1 | 482.642 | 482.67 | 486.72 | 487.053 | 0 |
| 640 | lancedb | bge_base_en | VECTOR_L2 | 2.1 | 482.681 | 483.026 | 485.576 | 485.898 | 0 |
| 641 | lancedb | bge_base_en | VECTOR_cosine | 2.1 | 482.695 | 483.038 | 485.179 | 486.106 | 0 |
| 642 | lancedb | e5_base_base | VECTOR_cosine | 2.1 | 482.98 | 481.743 | 487.265 | 487.7 | 0 |
| 643 | lancedb | snowflake_arctic_l | VECTOR_default | 2.1 | 483.384 | 480.454 | 498.649 | 507.303 | 0 |
| 644 | lancedb | mpnet_base | VECTOR_L2 | 2.1 | 483.478 | 481.526 | 496.296 | 504.26 | 0 |
| 645 | lancedb | snowflake_arctic_l | VECTOR_cosine | 2.1 | 483.548 | 482.91 | 486.678 | 488.318 | 0 |
| 646 | lancedb | bge_squad | VECTOR_default | 2.1 | 483.725 | 482.993 | 489.511 | 491.438 | 0 |
| 647 | lancedb | snowflake_arctic_l_base | VECTOR_default | 2.1 | 484.026 | 484.212 | 490.806 | 492.131 | 0 |
| 648 | lancedb | e5_small | VECTOR_limit50 | 2.1 | 484.48 | 484.724 | 490.879 | 491.569 | 0 |
| 649 | lancedb | snowflake_arctic_l | VECTOR_L2 | 2.1 | 484.617 | 483.734 | 491.891 | 496.608 | 0 |
| 650 | lancedb | mpnet_multi | VECTOR_limit20 | 2.1 | 484.933 | 485.42 | 486.721 | 487.188 | 0 |
| 651 | lancedb | distilroberta | VECTOR_limit20 | 2.1 | 485.476 | 486.234 | 487.443 | 487.475 | 0 |
| 652 | lancedb | snowflake_arctic_l_base | VECTOR_L2 | 2.1 | 485.544 | 485.177 | 488.378 | 490.001 | 0 |
| 653 | lancedb | bge_squad | VECTOR_L2 | 2.1 | 485.984 | 486.391 | 488.76 | 488.919 | 0 |
| 654 | lancedb | bge_squad | VECTOR_limit5 | 2.1 | 486.5 | 482.368 | 511.197 | 530.95 | 0 |
| 655 | lancedb | snowflake_arctic_l_base | VECTOR_cosine | 2.1 | 486.559 | 485.494 | 493.747 | 499.049 | 0 |
| 656 | lancedb | bge_squad | VECTOR_cosine | 2.1 | 486.746 | 486.142 | 490.111 | 490.289 | 0 |
| 657 | lancedb | all_mini_l6_base | VECTOR_limit50 | 2.1 | 487.054 | 487.651 | 489.573 | 489.867 | 0 |
| 658 | lancedb | mpnet_base | VECTOR_limit20 | 2.1 | 487.102 | 486.573 | 489.843 | 489.969 | 0 |
| 659 | lancedb | e5_base | VECTOR_limit20 | 2.1 | 487.47 | 483.632 | 506.7 | 508.452 | 0 |
| 660 | lancedb | mpnet_multi_base | VECTOR_limit20 | 2.1 | 487.632 | 485.254 | 500.842 | 512.489 | 0 |
| 661 | lancedb | gte_small | VECTOR_limit50 | 2 | 487.882 | 486.873 | 492.354 | 493.063 | 0 |
| 662 | lancedb | e5_base_hf | VECTOR_limit20 | 2 | 487.956 | 487.783 | 490.368 | 490.419 | 0 |
| 663 | lancedb | bge_base_en | VECTOR_limit20 | 2 | 488.118 | 488.164 | 491.321 | 492.506 | 0 |
| 664 | lancedb | multi_qa_mpnet | VECTOR_limit20 | 2 | 488.16 | 488.149 | 490.798 | 491.125 | 0 |
| 665 | lancedb | multi_qa_minilm | VECTOR_limit50 | 2 | 488.606 | 487.311 | 493.759 | 494.063 | 0 |
| 666 | lancedb | bge_squad_base | VECTOR_limit5 | 2 | 488.955 | 484.045 | 513.731 | 535.688 | 0 |
| 667 | lancedb | bge_squad_base | VECTOR_default | 2 | 489.064 | 487.771 | 494.12 | 495.562 | 0 |
| 668 | lancedb | e5_base_base | VECTOR_limit20 | 2 | 489.704 | 489.57 | 492.809 | 493.505 | 0 |
| 669 | lancedb | bge_squad_base | VECTOR_L2 | 2 | 489.716 | 489.033 | 497.4 | 498.936 | 0 |
| 670 | lancedb | snowflake_arctic_l | VECTOR_limit20 | 2 | 489.979 | 490.212 | 492.23 | 492.279 | 0 |
| 671 | lancedb | bge_squad_base | VECTOR_cosine | 2 | 490.528 | 490.461 | 494.459 | 496.8 | 0 |
| 672 | lancedb | snowflake_arctic_l_base | VECTOR_limit20 | 2 | 490.625 | 490.667 | 493.721 | 494.428 | 0 |
| 673 | lancedb | e5_base | VECTOR_cosine | 2 | 491.738 | 486.847 | 519.953 | 532.195 | 0 |
| 674 | lancedb | minilm_l12 | VECTOR_limit50 | 2 | 491.921 | 491.656 | 496.956 | 499.494 | 0 |
| 675 | lancedb | bge_squad | VECTOR_limit20 | 2 | 492 | 491.065 | 495.785 | 496.698 | 0 |
| 676 | lancedb | all_mini_l6 | VECTOR_limit50 | 2 | 492.323 | 491.769 | 495.891 | 497.041 | 0 |
| 677 | lancedb | bge_small_en | VECTOR_limit50 | 2 | 492.861 | 492.06 | 499.626 | 500.659 | 0 |
| 678 | lancedb | paraphrase_multi | VECTOR_limit50 | 2 | 493.425 | 493.1 | 499.265 | 500.485 | 0 |
| 679 | lancedb | e5_small_hf | VECTOR_limit50 | 2 | 494.406 | 493.815 | 501.014 | 502.24 | 0 |
| 680 | lancedb | bge_squad_base | VECTOR_limit20 | 2 | 495.585 | 495.213 | 500.044 | 500.216 | 0 |
| 681 | lancedb | gte_base | VECTOR_limit50 | 2 | 496.278 | 496.025 | 497.938 | 498.01 | 0 |
| 682 | lancedb | e5_small_base | VECTOR_limit50 | 2 | 496.571 | 496.372 | 502.76 | 503.707 | 0 |
| 683 | lancedb | distilroberta | VECTOR_limit50 | 2 | 499.849 | 500.029 | 502.197 | 502.275 | 0 |
| 684 | lancedb | mpnet_base | VECTOR_limit50 | 2 | 500.702 | 500.183 | 503.789 | 503.942 | 0 |
| 685 | lancedb | mpnet_multi | VECTOR_limit50 | 2 | 502.195 | 501.943 | 506.093 | 507.119 | 0 |
| 686 | lancedb | mpnet_multi_base | VECTOR_limit50 | 2 | 502.858 | 501.724 | 509.695 | 512.911 | 0 |
| 687 | lancedb | bge_base_en | VECTOR_limit50 | 2 | 505.104 | 505.304 | 509.303 | 509.489 | 0 |
| 688 | lancedb | snowflake_arctic_l | VECTOR_limit50 | 2 | 505.943 | 505.512 | 508.296 | 508.737 | 0 |
| 689 | lancedb | e5_base_hf | VECTOR_limit50 | 2 | 506.288 | 503.463 | 522.052 | 535.889 | 0 |
| 690 | lancedb | multi_qa_mpnet | VECTOR_limit50 | 2 | 507.223 | 506.924 | 511.044 | 512.445 | 0 |
| 691 | lancedb | bge_squad | VECTOR_limit50 | 2 | 507.507 | 507.683 | 510.1 | 510.311 | 0 |
| 692 | lancedb | e5_base_base | VECTOR_limit50 | 2 | 507.833 | 506.572 | 513.793 | 514.626 | 0 |
| 693 | lancedb | snowflake_arctic_l_base | VECTOR_limit50 | 2 | 511.25 | 507.903 | 526.351 | 536.662 | 0 |
| 694 | lancedb | e5_base | VECTOR_L2 | 2 | 512.725 | 476.241 | 675.952 | 780.022 | 0 |
| 695 | lancedb | bge_squad_base | VECTOR_limit50 | 1.9 | 514.504 | 513.41 | 519.605 | 521.983 | 0 |
| 696 | lancedb | e5_base | VECTOR_limit50 | 1.9 | 526.519 | 527.247 | 530.506 | 530.596 | 0 |

## Encode VRAM

_ENCODE FAZI VRAM PEAK (GB) — log HEARTBEAT'ten_  
| Model | encode (full corpus) | encode_qual (SQuAD) |
|---|---|---|
| e5_small | 0.49 | - |
| mpnet_multi | 1.13 | - |
| e5_base | 1.15 | 1.16 |
| bge_squad | 1.39 | 1.41 |
| qwen_lora | - | - |
| snowflake_arctic_l | 1.24 | 1.17 |
| all_mini_l6 | 0.06 | - |
| bge-m3-fine | 2.54 | - |
| e5_small_base | 0.49 | - |
| mpnet_multi_base | 1.13 | - |
| e5_base_base | 1.15 | 1.16 |
| bge_squad_base | 1.49 | 1.42 |
| qwen_lora_base | 2.03 | - |
| snowflake_arctic_l_base | 2.32 | 2.34 |
| all_mini_l6_base | 0.12 | - |
| bge_m3_base | - | - |
| minilm_l12 | 0.15 | - |
| mpnet_base | 0.49 | 0.49 |
| distilroberta | 0.36 | 0.35 |
| multi_qa_minilm | 0.12 | - |
| multi_qa_mpnet | 0.48 | 0.49 |
| paraphrase_multi | 0.49 | - |
| bge_small_en | 0.16 | - |
| bge_base_en | 0.47 | 0.5 |
| gte_small | 0.08 | - |
| gte_base | 0.25 | - |
| e5_small_hf | 0.15 | - |
| e5_base_hf | 0.56 | 0.5 |
