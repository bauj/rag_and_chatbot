"""
Download local models to the HuggingFace cache (~/.cache/huggingface/hub).

Run this once before using the chatbot with local models:
    python download_models.py

Models downloaded:
- sentence-transformers/all-MiniLM-L6-v2  (embedding)
- BAAI/bge-reranker-v2-m3                 (reranker, optional)
"""

from sentence_transformers import SentenceTransformer, CrossEncoder

print("Downloading embedding model (all-MiniLM-L6-v2)...")
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("  Done.")

print("Downloading reranker model (BAAI/bge-reranker-v2-m3)...")
CrossEncoder('BAAI/bge-reranker-v2-m3')
print("  Done.")

print("\nAll models downloaded.")
