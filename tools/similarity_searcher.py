import os
import re
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def embed_texts(texts: List[str], model: SentenceTransformer):
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

def tool_similarity_search(
    query: str,
    folder_path: str,
    threshold: float = 0.35,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Dict[str, Tuple[bool, float]]:
    """
    Returns dictionary mapping file_path -> (is_relevant, similarity_score)
    """

    model = SentenceTransformer(model_name)
    file_scores = {}

    # Load and embed query
    query_embedding = embed_texts([query], model)[0].cpu().detach().numpy()

    # Read all files and compute similarity
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        content = extract_text_from_file(file_path)
        cleaned = clean_text(content)
        if not cleaned:
            continue

        file_embedding = embed_texts([cleaned], model)[0].cpu().detach().numpy()
        sim = cosine_similarity([query_embedding], [file_embedding])[0][0]
        file_scores[file_path] = (sim >= threshold, sim)

    return file_scores
