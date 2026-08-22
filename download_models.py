"""
Download local models to the HuggingFace cache (~/.cache/huggingface/hub).

Run this once before using the chatbot with local models:
    python download_models.py

Models downloaded:
- sentence-transformers/all-MiniLM-L6-v2   (embedding)
- BAAI/bge-reranker-v2-m3                  (reranker, reranker.type: "cross_encoder", default)
- answerdotai/answerai-colbert-small-v1    (reranker, reranker.type: "late_interaction", optional)
"""

from sentence_transformers import SentenceTransformer, CrossEncoder, MultiVectorEncoder

print("Downloading embedding model (all-MiniLM-L6-v2)...")
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("  Done.")

print("Downloading reranker model (BAAI/bge-reranker-v2-m3)...")
CrossEncoder('BAAI/bge-reranker-v2-m3')
print("  Done.")

print("Downloading late-interaction reranker model (answerai-colbert-small-v1)...")
MultiVectorEncoder('answerdotai/answerai-colbert-small-v1')
print("  Done.")

print("\nAll models downloaded.")
