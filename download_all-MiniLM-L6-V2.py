import os
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

project_cache = './models'
os.makedirs(project_cache, exist_ok=True)

print("Downloading model...")
_ = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    cache_folder=project_cache
)